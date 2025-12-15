/**
 * Navigation Suite Runner
 *
 * Run this to execute all navigation experiments and see if learning works.
 *
 * Usage: bun run dist/src/suites/navigation/run.mjs
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { createNavigationSuite } from './suite.ts'

async function main() {
  console.log('╔════════════════════════════════════════════════════════════════╗')
  console.log('║           VESSEL - Navigation Experiment Suite                  ║')
  console.log('║                                                                 ║')
  console.log('║   HYPOTHESIS: Creatures can learn to navigate gradients         ║')
  console.log('║                                                                 ║')
  console.log('║   Experiments:                                                  ║')
  console.log('║   01. Random baseline (expect: FAIL)                            ║')
  console.log('║   02. Innate wiring only (expect: BASIC SUCCESS)                ║')
  console.log('║   03. Learning enabled (expect: UNKNOWN - real test!)           ║')
  console.log('╚════════════════════════════════════════════════════════════════╝')
  console.log('')

  const suite = createNavigationSuite()

  // Run with 1000 timesteps per experiment
  // This is ~1 second of simulated time per experiment
  const TIMESTEPS = 1000

  console.log(`Running ${suite.experiments.length} experiments, ${TIMESTEPS} timesteps each...`)
  console.log('')

  const startTime = Date.now()

  try {
    const result = await suite.runAll(TIMESTEPS)

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)

    console.log('')
    console.log('╔════════════════════════════════════════════════════════════════╗')
    console.log('║                    SUITE RESULTS                               ║')
    console.log('╚════════════════════════════════════════════════════════════════╝')
    console.log('')

    // Print individual results
    console.log('INDIVIDUAL EXPERIMENTS:')
    console.log('-'.repeat(60))

    for (const [name, expResult] of result.results) {
      const analysis = result.analyses.get(name)
      console.log(`  ${name}:`)
      console.log(`    Cumulative Reward: ${expResult.finalCumulativeReward.toFixed(4)}`)
      console.log(`    Learning Detected: ${analysis?.learningDetected ?? 'N/A'}`)
      console.log(`    Reward Trend: ${analysis?.rewardTrend ?? 'N/A'}`)
      console.log('')
    }

    // Print comparisons
    console.log('COMPARISONS:')
    console.log('-'.repeat(60))

    for (const comparison of result.comparisons) {
      console.log(`  ${comparison.baseline.experimentName} vs ${comparison.treatment.experimentName}:`)
      console.log(`    Difference: ${comparison.rewardDifference.toFixed(4)}`)
      console.log(`    Improvement: ${comparison.percentImprovement.toFixed(1)}%`)
      console.log(`    Significant: ${comparison.statisticalSignificance}`)
      console.log(`    ${comparison.conclusion}`)
      console.log('')
    }

    // Print conclusion
    console.log('═'.repeat(64))
    console.log('')
    console.log(`HYPOTHESIS SUPPORTED: ${result.hypothesisSupported}`)
    console.log(`CONCLUSION: ${result.conclusion}`)
    console.log('')

    // Print The Slap
    console.log(result.suiteSlap)

    console.log('')
    console.log(`Total time: ${elapsed}s`)
    console.log('')

    // Final verdict
    if (result.hypothesisSupported === true) {
      console.log('🎉 LEARNING WORKS! The creature learned to navigate!')
    } else if (result.hypothesisSupported === false) {
      console.log('❌ Learning did not improve navigation. Back to the drawing board.')
    } else {
      console.log('🤔 Results inconclusive. More investigation needed.')
    }

  } catch (error) {
    console.error('Experiment failed:', error)
    process.exit(1)
  }
}

main().catch(console.error)
