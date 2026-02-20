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

#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a chunk of bytes and returns any events it completed.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for &byte in chunk {
            if byte == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if let Some(event) = self.process_line(&line) {
                    events.push(event);
                }
            } else {
                self.line.push(byte);
            }
        }
        events
    }

    /// Call when the body ends. Returns a final event if the stream stopped
    /// without the trailing blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            if let Some(event) = self.process_line(&line) {
                return Some(event);
            }
        }
        self.dispatch()
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
        let mut parser = SseParser::new();
        let mut events = parser.push(input);
        events.extend(parser.finish());
        events
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
                events.extend(parser.push(chunk));
            }
            events.extend(parser.finish());
            assert_eq!(events, expected, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn finish_flushes_an_unterminated_event() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: partial").is_empty());
        assert_eq!(parser.finish(), Some(ev(None, "partial")));
        assert_eq!(parser.finish(), None);
    }

    #[test]
    fn encode_round_trips() {
        let encoded = encode_data("a\nb");
        assert_eq!(&encoded[..], b"data: a\ndata: b\n\n");
        assert_eq!(parse_all(&encoded), vec![ev(None, "a\nb")]);
    }
}
