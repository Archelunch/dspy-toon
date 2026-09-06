// Run pinned upstream generators and scoring without changing their implementation.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const [mode, root, input, output] = process.argv.slice(2);
const load = p => import(pathToFileURL(`${root}/${p}`).href);
const { encode, decode } = await load('packages/toon/src/index.ts');
if (mode === 'export') {
  const { ACCURACY_DATASETS } = await load('benchmarks/src/datasets.ts');
  const { generateQuestions } = await load('benchmarks/src/questions/index.ts');
  const { FORMATS, supportsCSV } = await load('benchmarks/src/formats.ts');
  const { encodeDataset } = await load('benchmarks/src/structural-corruption.ts');
  const cases = generateQuestions().map(q => {
    const dataset = ACCURACY_DATASETS.find(d => d.name === q.dataset);
    const formats = {};
    for (const name of ['json-pretty', 'json-compact', 'toon', 'csv']) {
      if (name === 'csv' && !supportsCSV(dataset)) continue;
      const f = FORMATS[name];
      formats[name] = {text: encodeDataset(f, dataset), primer: f.primer, fence: f.fence};
    }
    if (!dataset.corruption) assert.deepEqual(decode(encode(dataset.data)), dataset.data);
    return {...q, data: dataset.data, metadata: dataset.metadata, corruption: dataset.corruption, formats};
  });
  fs.writeFileSync(output, JSON.stringify(cases));
  console.log(`Exported ${cases.length} upstream questions`);
} else if (mode === 'verify') {
  const cases = JSON.parse(fs.readFileSync(input));
  let identical = 0;
  for (const c of cases) {
    assert.deepEqual(decode(c.toon), c.data);
    assert.deepEqual(decode(encode(c.data)), c.data);
    if (encode(c.data) === c.toon) identical++;
  }
  fs.writeFileSync(output, JSON.stringify({cases:cases.length, reference_decode_equal:cases.length, byte_identical:identical}));
} else if (mode === 'score') {
  const { compareAnswers } = await load('benchmarks/src/normalize.ts');
  const rows = JSON.parse(fs.readFileSync(input));
  fs.writeFileSync(output, JSON.stringify(rows.map(r => ({job_id:r.job_id,...compareAnswers(r.actual,r.expected,r.kind,r.options)}))));
} else throw new Error('Unknown mode');
