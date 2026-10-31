import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from racer.validator import validate
from racer.generator import OpenAICompatBackend, AnthropicBackend


def test_substring_names_and_digits_in_names():
    ctx = {"tier": "LOW", "products": [{"name": "Synthetic Fund 12", "risk_level": 1.0}], "clauses": []}
    names = ["Synthetic Fund 1", "Synthetic Fund 12", "Synthetic Fund 120"]
    r = validate("Synthetic Fund 12 [P1] has risk level 1.", ctx, names)
    assert r.ok and r.metrics["product_grounding"] == 1.0 and r.metrics["numeric_grounding"] == 1.0


def test_generic_phrase_not_flagged_but_real_outside_product_is():
    ctx = {"tier": "LOW", "products": [{"name": "Alpha Debt Fund", "risk_level": 1.0}], "clauses": []}
    assert validate("Alpha Debt Fund [P1]. This fund is stable.", ctx, ["Alpha Debt Fund", "Beta Debt Fund"]).ok
    r = validate("Alpha Debt Fund [P1] and Beta Debt Fund.", ctx, ["Alpha Debt Fund", "Beta Debt Fund"])
    assert not r.ok and r.metrics["product_grounding"] == 0.5


class _H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/chat/completions"):
            out = {"choices": [{"message": {"content": "openai:" + body["messages"][0]["content"][:5]}}]}
        else:
            assert self.headers["x-api-key"] == "k"
            out = {"content": [{"type": "text", "text": "anth:"}, {"type": "text", "text": body["messages"][0]["content"][:3]}]}
        b = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass


def test_http_backends_parse_responses():
    srv = HTTPServer(("127.0.0.1", 0), _H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    assert OpenAICompatBackend(base + "/v1", "k", "m").generate("hello world", {}) == "openai:hello"
    assert AnthropicBackend("k", "m", base_url=base).generate("hello", {}) == "anth:hel"
    srv.shutdown()
