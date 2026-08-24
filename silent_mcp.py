import json
import os
import subprocess
import sys
import threading

# MCP resource methods Codex probes that most servers don't implement.
_INTERCEPTS = {
    "resources/list":           {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list":             {"prompts": []},
}

_stdout_lock = threading.Lock()

def _write_stdout(data: bytes) -> None:
    with _stdout_lock:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def _forward_child_stdout(child_stdout) -> None:
    """Thread: forward child process stdout → our stdout safely in binary chunks."""
    try:
        while True:
            # Using 4096 chunks instead of iterating over lines prevents stream deadlocks
            # for MCP servers that buffer heavily or emit large blocks without newlines.
            chunk = child_stdout.read(4096)
            if not chunk:
                break
            _write_stdout(chunk)
    except Exception as e:
        import datetime
        log_path = os.path.join(os.path.expanduser("~"), ".mcp_wrapper_error.log")
        with open(log_path, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Forward thread error: {e}\n")


def _read_client_message(stream):
    """Read one JSONL or Content-Length-framed MCP message without truncation."""
    first_line = stream.readline()
    if not first_line:
        return None
    if not first_line.lower().startswith(b"content-length:"):
        return first_line, first_line, False

    headers = [first_line]
    try:
        content_length = int(first_line.split(b":", 1)[1].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid Content-Length header") from exc
    if content_length < 0:
        raise ValueError("Invalid negative Content-Length")

    while True:
        header_line = stream.readline()
        if not header_line:
            raise EOFError("Incomplete MCP headers")
        headers.append(header_line)
        if header_line in {b"\r\n", b"\n"}:
            break

    body = stream.read(content_length)
    if len(body) != content_length:
        raise EOFError("Incomplete MCP body")
    return b"".join(headers) + body, body, True


def _encode_intercept_response(message: dict, *, framed: bool) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    if framed:
        return (
            b"Content-Length: "
            + str(len(payload)).encode("ascii")
            + b"\r\n\r\n"
            + payload
        )
    return payload + b"\n"


def main():
    args = sys.argv[1:]
    if not args:
        return

    # Restrict interception mode strictly to callers that request it
    codex_mode = False
    if args[0] == "--codex":
        codex_mode = True
        args = args[1:]
        if not args:
            return

    # Suppress console window on Windows
    creationflags = 0x08000000 if os.name == "nt" else 0

    # Windows requires 'npx.cmd' or 'npm.cmd' instead of bare 'npx' if shell=False
    if os.name == "nt":
        arg0 = args[0].lower()
        if arg0 == "npx":
            args[0] = "npx.cmd"
        elif arg0 == "npm":
            args[0] = "npm.cmd"

    # Under pythonw.exe, sys.stderr is None (no console); passing None to Popen
    # causes CreateProcess to receive a null stderr handle which, combined with
    # CREATE_NO_WINDOW + PIPE stdin/stdout, triggers [Errno 22] Invalid argument.
    stderr_target = sys.stderr if sys.stderr is not None else subprocess.DEVNULL

    # If stdin/stdout is dead on launch, gracefully exit to prevent hanging infinite loops.
    if sys.stdin is None or sys.stdout is None:
        import datetime
        log_path = os.path.join(os.path.expanduser("~"), ".mcp_wrapper_error.log")
        with open(log_path, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Error: sys.stdin or sys.stdout is None\n")
        sys.exit(1)

    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            creationflags=creationflags,
            bufsize=0,
        )

        fwd = threading.Thread(
            target=_forward_child_stdout,
            args=(proc.stdout,),
            daemon=True,
        )
        fwd.start()

        # Read framing-aware MCP messages. Content-Length bodies need not end in
        # a newline, so line iteration can otherwise stall or corrupt requests.
        while True:
            parsed_message = _read_client_message(sys.stdin.buffer)
            if parsed_message is None:
                break
            raw_message, payload, is_framed = parsed_message
            intercepted = False

            if codex_mode:
                try:
                    msg = json.loads(payload)
                    method = msg.get("method", "")
                    msg_id = msg.get("id")

                    if method in _INTERCEPTS and msg_id is not None:
                        # Respond directly with a valid empty result, protecting Codex
                        resp = _encode_intercept_response(
                            {
                                "jsonrpc": "2.0",
                                "id": msg_id,
                                "result": _INTERCEPTS[method],
                            },
                            framed=is_framed,
                        )
                        _write_stdout(resp)
                        intercepted = True
                except (ValueError, KeyError):
                    pass  # Fall through

            if not intercepted:
                try:
                    proc.stdin.write(raw_message)
                    proc.stdin.flush()
                except BrokenPipeError:
                    break

        proc.stdin.close()
        proc.wait()
        sys.exit(proc.returncode)

    except Exception as e:
        import datetime
        log_path = os.path.join(os.path.expanduser("~"), ".mcp_wrapper_error.log")
        with open(log_path, "a") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Error launching {args}: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
