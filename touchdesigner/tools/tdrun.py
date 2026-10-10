"""Run a Python file (or inline code) inside my TouchDesigner instance (port 9990) through the MCP web server.
usage:  python tdrun.py file.py        -> executes the file in TD
        python tdrun.py -c "code"      -> executes inline
The script may set `result = ...`; stdout / stderr are echoed. NOTE: the bridge re-executes the LAST line of a script,
so end scripts with a harmless assignment (x = 1).
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:9990/api/td/server/exec"


def run(code, timeout=600):
    req = urllib.request.Request(URL, data=json.dumps({"script": code}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv):
    code = argv[1] if argv[0] == "-c" else open(argv[0], encoding="utf-8").read()
    if argv[0] == "-c":
        code = argv[1]
    try:
        out = run(code)
    except Exception as e:
        print("HTTP ERROR", e)
        return 2
    d = out.get("data") or {}
    if d.get("stdout"):
        print(d["stdout"], end="")
    if d.get("stderr"):
        print("STDERR:", d["stderr"], end="")
    if out.get("error"):
        print("ERROR:", out["error"])
    if d.get("result") is not None:
        print("RESULT:", json.dumps(d["result"], indent=1, default=str)[:60000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
