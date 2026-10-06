// Invoke the maintained runtime controls from the exact linked FastSecDec checkout.
import assert from "node:assert/strict";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const checkout = process.env.FASTSECDEC_CHECKOUT;
assert(checkout, "Set FASTSECDEC_CHECKOUT to the FastSecDec checkout resolved by Cargo for this wheel");
process.argv[3] = fileURLToPath(new URL("../../", import.meta.url));
await import(pathToFileURL(join(checkout, "bindings/python/scripts/test-pyodide.mjs")));
