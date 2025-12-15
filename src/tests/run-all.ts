/**
 * Test Runner - Brain Primitives Complete Test Suite
 *
 * Runs all tests in dependency order:
 * 1. Neuron dynamics (foundation)
 * 2. Synapse transmission (builds on neurons)
 * 3. STDP learning (builds on synapses)
 * 4. Neuromodulation (gates learning)
 * 5. Genome system (orchestrates all)
 * 6. Sensory-Motor pathway (end-to-end)
 *
 * The Slap Protocol: Every test must prove something real.
 * No fake successes. No cherry-picking. No pretending.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { printFinalReport, getTotalResults } from './utils.ts'
import { runNeuronTests } from './01-neuron.ts'
import { runSynapseTests } from './02-synapse.ts'
import { runSTDPTests } from './03-stdp.ts'
import { runModulationTests } from './04-modulation.ts'
import { runGenomeTests } from './05-genome.ts'
import { runSensoryMotorTests } from './06-sensory-motor.ts'

console.log('╔════════════════════════════════════════════════════════════════╗')
console.log('║         VESSEL - Brain Primitives Test Suite                   ║')
console.log('║                                                                 ║')
console.log('║  "The Slap Protocol"                                           ║')
console.log('║  Every test must prove something REAL.                         ║')
console.log('║  No fake successes. No cherry-picking. No pretending.          ║')
console.log('╚════════════════════════════════════════════════════════════════╝')
console.log('')

const startTime = Date.now()

// Run all test suites in dependency order
runNeuronTests()       // Foundation: Do neurons work?
runSynapseTests()      // Layer 1: Does transmission work?
runSTDPTests()         // Layer 2: Does learning work?
runModulationTests()   // Layer 3: Does modulation gate learning?
runGenomeTests()       // Layer 4: Does DNA create networks?
runSensoryMotorTests() // Layer 5: Does stimulus cause response?

const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)

// Final report
printFinalReport()

console.log(`\nTotal time: ${elapsed}s`)
console.log('')

// Exit with appropriate code
const { passed, failed } = getTotalResults()
if (failed > 0) {
  console.log(`\n❌ ${failed} test(s) failed. Fix before creating creatures.\n`)
  process.exit(1)
} else {
  console.log(`\n✅ All ${passed} tests passed! Ready to create creatures.\n`)
  process.exit(0)
}
