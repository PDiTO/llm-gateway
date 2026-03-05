//! Incremental Server-Sent Events parser and encoder.
//!
//! Upstream bodies arrive in arbitrary byte chunks, so an event (or a line, or
//! a UTF-8 character) can be split across chunks. The parser buffers partial
//! lines and only hands out complete events.

use bytes::Bytes;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if the server sent one. Anthropic always does,
    /// OpenAI never does.
    pub event: Option<String>,
    /// All `data:` lines of the event joined with `\n`.
    pub data: String,
}

/// Upper bound on a single event, so a misbehaving upstream that never sends
/// a newline can't make the gateway buffer without limit.
pub const DEFAULT_MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("upstream sent an SSE event larger than {0} bytes")]
pub struct EventTooLarge(pub usize);

#[derive(Debug)]
pub struct SseParser {
    line: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
    max_event_bytes: usize,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::with_max_event_bytes(DEFAULT_MAX_EVENT_BYTES)
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_event_bytes(max_event_bytes: usize) -> Self {
        Self {
            line: Vec::new(),
            event: None,
            data: String::new(),
            has_data: false,
            max_event_bytes,
        }
    }

    /// Feeds a chunk of bytes and returns any events it completed.
    ///
    /// When the body ends, whatever is still buffered is an incomplete event.
    /// The spec says to discard it, so there is no `finish` step: a stream cut
    /// off mid-event never produces half an event.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, EventTooLarge> {
        let mut events = Vec::new();
        let mut rest = chunk;
        while let Some(newline) = rest.iter().position(|&b| b == b'\n') {
            self.line.extend_from_slice(&rest[..newline]);
            rest = &rest[newline + 1..];
            let mut line = std::mem::take(&mut self.line);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.process_line(&line) {
                events.push(event);
            }
            self.check_size()?;
        }
        self.line.extend_from_slice(rest);
        self.check_size()?;
        Ok(events)
    }

    fn check_size(&self) -> Result<(), EventTooLarge> {
        if self.line.len() + self.data.len() > self.max_event_bytes {
            Err(EventTooLarge(self.max_event_bytes))
        } else {
            Ok(())
        }
    }

    fn process_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line[0] == b':' {
            return None; // comment / keep-alive
        }
        let line = String::from_utf8_lossy(line);
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line.as_ref(), ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            // `id` and `retry` don't matter for a one-shot proxy stream.
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        if !self.has_data {
            return None;
        }
        self.has_data = false;
        Some(SseEvent {
            event,
            data: std::mem::take(&mut self.data),
        })
    }
}

/// Encodes a `data:` event. Payloads containing newlines are split across
/// several `data:` lines, as the spec requires.
pub fn encode_data(payload: &str) -> Bytes {
    let mut out = String::with_capacity(payload.len() + 8);
    for line in payload.split('\n') {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    Bytes::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_all(input: &[u8]) -> Vec<SseEvent> {
        SseParser::new().push(input).unwrap()
    }

    fn ev(event: Option<&str>, data: &str) -> SseEvent {
        SseEvent {
            event: event.map(str::to_string),
            data: data.to_string(),
        }
    }

    #[test]
    fn parses_named_and_unnamed_events() {
        let input =
            b"event: ping\ndata: {\"type\":\"ping\"}\n\ndata: {\"a\":1}\n\ndata: [DONE]\n\n";
        assert_eq!(
            parse_all(input),
            vec![
                ev(Some("ping"), r#"{"type":"ping"}"#),
                ev(None, r#"{"a":1}"#),
                ev(None, "[DONE]"),
            ]
        );
    }

    #[test]
    fn handles_crlf_comments_and_multiline_data() {
        let input = b": keep-alive\r\ndata: line one\r\ndata: line two\r\nid: 7\r\n\r\n";
        assert_eq!(parse_all(input), vec![ev(None, "line one\nline two")]);
    }

    #[test]
    fn value_without_space_and_field_without_colon() {
        let input = b"data:tight\n\ndata\n\n";
        assert_eq!(parse_all(input), vec![ev(None, "tight"), ev(None, "")]);
    }

    #[test]
    fn event_without_data_is_dropped() {
        assert_eq!(
            parse_all(b"event: lonely\n\ndata: x\n\n"),
            vec![ev(None, "x")]
        );
    }

    #[test]
    fn same_events_whatever_the_chunking() {
        let input = "event: content_block_delta\ndata: {\"text\":\"héllo 🦀\"}\n\nevent: message_stop\ndata: {}\n\n";
        let expected = parse_all(input.as_bytes());
        assert_eq!(expected.len(), 2);
        for chunk_size in 1..input.len() {
            let mut parser = SseParser::new();
            let mut events = Vec::new();
            for chunk in input.as_bytes().chunks(chunk_size) {
                events.extend(parser.push(chunk).unwrap());
            }
            assert_eq!(events, expected, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn unterminated_event_is_never_dispatched() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: {\"partial\": tr").unwrap().is_empty());
        assert!(parser.push(b"ue}\n").unwrap().is_empty());
        // Only the blank line completes it.
        assert_eq!(
            parser.push(b"\n").unwrap(),
            vec![ev(None, r#"{"partial": true}"#)]
        );
    }

    #[test]
    fn oversized_events_are_rejected() {
        let mut parser = SseParser::with_max_event_bytes(16);
        assert!(parser.push(b"data: short\n\n").is_ok());
        assert_eq!(parser.push(&[b'x'; 17]), Err(EventTooLarge(16)));

        let mut parser = SseParser::with_max_event_bytes(16);
        let many_lines = b"data: 0123456789\ndata: 0123456789\n";
        assert!(parser.push(many_lines).is_err());
    }

    #[test]
    fn encode_round_trips() {
        let encoded = encode_data("a\nb");
        assert_eq!(&encoded[..], b"data: a\ndata: b\n\n");
        assert_eq!(parse_all(&encoded), vec![ev(None, "a\nb")]);
    }
}
