// Operator-triggered native runner probe, not a Gemini model/chat lifecycle.
// Input stays in pipes; only fixed nonsecret result fields reach stdout.
import {pathToFileURL} from 'node:url';
let input = '';
for await (const part of process.stdin) input += part;
try {
  const probe = JSON.parse(input);
  globalThis.fetch = async () => {throw new Error('probe_network_disabled');};
  const {HookRunner} = await import(pathToFileURL(probe.core).href);
  const config = {sanitizationConfig: {}, storage: {getPlansDir: () => probe.cwd}};
  const runner = new HookRunner(config);
  for (const event of ['AfterAgent', 'PreCompress']) {
    const result = await runner.executeHook({...probe.hooks[event], source: 'user'}, event, {
      hook_event_name: event, cwd: probe.cwd, session_id: 'muninn-operator-capture-probe',
      transcript_path: probe.transcript, trigger: 'manual', prompt: '', prompt_response: '',
      stop_hook_active: false,
    });
    process.stdout.write(JSON.stringify({stage: 'native_runner', event,
      success: result.success === true, duration_ms: result.duration,
      exit_code: result.exitCode ?? null, diagnostic_output_present: Boolean(result.stderr)}) + '\n');
  }
} catch (error) {
  process.stdout.write(JSON.stringify({stage: 'native_runner', success: false,
    error_code: 'installed_runner_unavailable', error_type:
      ['TypeError', 'SyntaxError', 'ReferenceError', 'Error'].includes(error?.name) ? error.name : 'unavailable'}) + '\n');
  process.exitCode = 1;
}
