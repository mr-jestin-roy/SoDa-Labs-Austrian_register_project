"""bench_reich.py — G3_reich condition: G1 structure + every entry must START with one of the
82,060 Reich-wide names (OCR-cleaned KG v2 surnames seen >=2 times outside the bench pages +
frequent organisation starts). Same cells, prompt, runs and timing protocol as bench.py.
Writes /mnt/work/bench_xgrammar/reich/runs.jsonl and wall.json."""
import importlib.util, sys, time
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

names = [n for n in (HERE / "reich_names.txt").read_text(encoding="utf-8").splitlines() if n]
alts = "\n    | ".join(f'"{bench.esc(n)}"' for n in names)
G_REICH = bench.G1.replace('entry ::= ext " " text', 'entry ::= ext " " name rest') + \
    f"name ::= {alts}\nrest ::= [^\\n]*\n"

if __name__ == "__main__":
    if "--compile-only" in sys.argv:
        import xgrammar as xg
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        comp = xg.GrammarCompiler(xg.TokenizerInfo.from_huggingface(tok))
        t0 = time.time(); cg = comp.compile_grammar(G_REICH)
        print(f"names={len(names)} grammar_chars={len(G_REICH)} compile_s={time.time() - t0:.1f} "
              f"memory_MB={cg.memory_size_bytes / 1e6:.1f}")
        m = xg.GrammarMatcher(cg)
        for s in ["1240 Müller, Karl\n15 Bahnhof Pernegg\n", "7 Dirnbacher, Jakob\n"]:
            ok = all(m.accept_string(ch) for ch in [s]); print(repr(s[:30]), "accepted" if ok else "REJECTED"); m.reset()
        sys.exit()
    bench.GRAMMARS = {"G3_reich": lambda p: G_REICH}
    bench.WORK = Path("/mnt/work/bench_xgrammar/reich"); bench.WORK.mkdir(parents=True, exist_ok=True)
    bench.main()
