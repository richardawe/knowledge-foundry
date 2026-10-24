// Calling the foundry from Requstory's research phase. No dependencies: Node 18+
// (global fetch) or any runtime with fetch.
//
// Two ways to run it; pick one per environment:
//   * HTTP: `foundry serve-research` (or the Docker image) next to the worker.
//   * CLI:  `python -m foundry.cli research request.json --out result.json`.
import { spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CONTRACT, type Job, type ResearchRequest, type ResearchResult } from "./types";

export interface HttpOptions {
  baseUrl: string;            // e.g. http://foundry:8100
  token?: string;             // FOUNDRY_API_TOKEN on the service
  pollMs?: number;            // default 3000
  timeoutMs?: number;         // default 10 minutes; a run is ~40 s per company
}

export async function researchOverHttp(request: ResearchRequest, opts: HttpOptions): Promise<ResearchResult> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (opts.token) headers.Authorization = `Bearer ${opts.token}`;
  const created = await fetch(`${opts.baseUrl}/v1/research`, {
    method: "POST", headers, body: JSON.stringify({ ...request, contract: CONTRACT }),
  });
  if (created.status !== 202) throw new Error(`foundry refused the request: ${created.status} ${await created.text()}`);
  const { status_url } = (await created.json()) as { status_url: string };

  const deadline = Date.now() + (opts.timeoutMs ?? 10 * 60_000);
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, opts.pollMs ?? 3000));
    const job = (await (await fetch(`${opts.baseUrl}${status_url}`, { headers })).json()) as Job;
    if (job.status === "done") return job.result;
    if (job.status === "failed") throw new Error(`foundry research failed: ${job.error}`);
  }
  throw new Error("foundry research timed out");
}

export interface CliOptions {
  python?: string;            // default "python3"
  cwd: string;                // a knowledge-foundry checkout (or where it is pip-installed)
  env?: NodeJS.ProcessEnv;    // FOUNDRY_CONTACT_EMAIL, optionally FOUNDRY_VERIFY_PROVIDER/MODEL
}

export async function researchOverCli(request: ResearchRequest, opts: CliOptions): Promise<ResearchResult> {
  const dir = await mkdtemp(join(tmpdir(), "foundry-"));
  const input = join(dir, "request.json");
  const output = join(dir, "result.json");
  await writeFile(input, JSON.stringify({ ...request, contract: CONTRACT }));
  await new Promise<void>((resolve, reject) => {
    const child = spawn(opts.python ?? "python3", ["-m", "foundry.cli", "research", input, "--out", output],
      { cwd: opts.cwd, env: { ...process.env, ...opts.env }, stdio: ["ignore", "inherit", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += d));
    child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`foundry exited ${code}: ${stderr}`))));
  });
  return JSON.parse(await readFile(output, "utf8")) as ResearchResult;
}
