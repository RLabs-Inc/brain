/**
 * Navigation Suite - Can Creatures Learn to Navigate?
 *
 * HYPOTHESIS: A creature with biologically-inspired wiring (C. elegans)
 * can learn to navigate a chemical gradient better than:
 * 1. Random wiring
 * 2. Innate wiring alone (without learning)
 *
 * This suite tests the core question: Does STDP + reward actually
 * produce emergent learning behavior?
 *
 * THE SLAP QUESTIONS FOR THIS SUITE:
 * 1. Did we hard-code navigation into the innate wiring?
 *    - Answer: We wired food→turn (local search), but NOT gradient-following
 * 2. Did we design the test knowing the answer?
 *    - Answer: We genuinely don't know if learning will improve on innate
 * 3. Is this learning, or just our wiring?
 *    - Answer: Compare innate vs learned - if learned is better, it's learning
 * 4. Could random weights do this?
 *    - Answer: That's experiment 01 - the baseline
 * 5. Would this survive scrutiny?
 *    - Answer: We're publishing all results, pass or fail
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Suite, Experiment, SuiteResult, Comparison } from '../types.ts'
import { compareResults } from '../runner.svelte.ts'
import { createRandomExperiment } from './01-random.ts'
import { createInnateExperiment } from './02-innate.ts'
import { createLearnedExperiment } from './03-learned.ts'

// ============================================================================
// SUITE DEFINITION
// ============================================================================

export const SUITE_NAME = 'Navigation'
export const SUITE_HYPOTHESIS = 'Creatures with STDP learning can navigate a chemical gradient better than random or innate-only wiring'

/**
 * Create the navigation experiment suite.
 */
export function createNavigationSuite(): Suite {
  const experiments: Experiment[] = [
    createRandomExperiment(),
    createInnateExperiment(),
    createLearnedExperiment(),
  ]

  async function runAll(
    timestepsPerExperiment: number,
    onExperimentComplete?: (exp: Experiment, result: any) => void
  ): Promise<SuiteResult> {
    const startTime = new Date()
    const results = new Map<string, any>()
    const analyses = new Map<string, any>()

    for (const experiment of experiments) {
      console.log(`\n${'='.repeat(60)}`)
      console.log(`EXPERIMENT: ${experiment.name}`)
      console.log(`Hypothesis: ${experiment.hypothesis}`)
      console.log(`Expected: ${experiment.expectedOutcome}`)
      console.log('='.repeat(60))

      // Setup the experiment (creates creature and world)
      experiment.setup()

      const result = await experiment.run(timestepsPerExperiment)
      results.set(experiment.name, result)

      const analysis = experiment.analyze(result)
      analyses.set(experiment.name, analysis)

      console.log(`\nResult: ${result.finalCumulativeReward.toFixed(4)} cumulative reward`)
      console.log(`Learning detected: ${analysis.learningDetected}`)

      onExperimentComplete?.(experiment, result)

      experiment.teardown()
    }

    // Generate comparisons
    const comparisons: Comparison[] = []
    const randomResult = results.get('01-random')
    const innateResult = results.get('02-innate')
    const learnedResult = results.get('03-learned')

    if (randomResult && innateResult) {
      comparisons.push(compareResults(randomResult, innateResult))
    }
    if (innateResult && learnedResult) {
      comparisons.push(compareResults(innateResult, learnedResult))
    }
    if (randomResult && learnedResult) {
      comparisons.push(compareResults(randomResult, learnedResult))
    }

    // Determine hypothesis support
    const innateAnalysis = analyses.get('02-innate')
    const learnedAnalysis = analyses.get('03-learned')

    // Hypothesis is supported if:
    // 1. Innate is better than random (proves wiring matters)
    // 2. Learned is better than innate (proves learning helps)
    const innateVsRandom = comparisons.find(c =>
      c.baseline.experimentName === '01-random' &&
      c.treatment.experimentName === '02-innate'
    )
    const learnedVsInnate = comparisons.find(c =>
      c.baseline.experimentName === '02-innate' &&
      c.treatment.experimentName === '03-learned'
    )

    let hypothesisSupported: boolean | 'inconclusive' = 'inconclusive'
    let conclusion = ''

    if (innateVsRandom && learnedVsInnate) {
      const innateWins = innateVsRandom.percentImprovement > 10
      const learningWins = learnedVsInnate.percentImprovement > 10

      if (innateWins && learningWins) {
        hypothesisSupported = true
        conclusion = 'SUPPORTED: Innate wiring helps, and learning improves on innate'
      } else if (innateWins && !learningWins) {
        hypothesisSupported = false
        conclusion = 'PARTIAL: Innate wiring helps, but learning does not improve on it'
      } else if (!innateWins && learningWins) {
        hypothesisSupported = 'inconclusive'
        conclusion = 'UNEXPECTED: Random wiring works as well as innate?!'
      } else {
        hypothesisSupported = false
        conclusion = 'FAILED: Neither innate wiring nor learning showed improvement'
      }
    }

    const endTime = new Date()

    // Generate suite-level Slap
    const suiteSlap = generateSuiteSlap(results, analyses, comparisons, conclusion)

    return {
      suiteName: SUITE_NAME,
      hypothesis: SUITE_HYPOTHESIS,
      startTime,
      endTime,
      results,
      analyses,
      comparisons,
      hypothesisSupported,
      conclusion,
      suiteSlap,
    }
  }

  function compare(baseline: any, treatment: any): Comparison {
    return compareResults(baseline, treatment)
  }

  return {
    name: SUITE_NAME,
    description: 'Tests whether creatures can learn to navigate chemical gradients',
    hypothesis: SUITE_HYPOTHESIS,
    experiments,
    runAll,
    compare,
  }
}

// ============================================================================
// SLAP GENERATION
// ============================================================================

function generateSuiteSlap(
  results: Map<string, any>,
  analyses: Map<string, any>,
  comparisons: Comparison[],
  conclusion: string
): string {
  const lines: string[] = [
    '',
    '╔════════════════════════════════════════════════════════════════╗',
    '║                    THE SLAP PROTOCOL                           ║',
    '║           Honesty Check for Navigation Suite                   ║',
    '╚════════════════════════════════════════════════════════════════╝',
    '',
    '1. DID WE HARD-CODE NAVIGATION?',
    '   - Innate wiring: food→turn (local search), NOT gradient-following',
    '   - If learned beats innate, gradient-following was LEARNED',
    '',
    '2. DID WE DESIGN THE TEST KNOWING THE ANSWER?',
    '   - We genuinely did not know if learning would help',
    '   - This was a real experiment, not a demonstration',
    '',
    '3. IS THIS LEARNING OR JUST WIRING?',
  ]

  const innateVsRandom = comparisons.find(c =>
    c.baseline.experimentName === '01-random' &&
    c.treatment.experimentName === '02-innate'
  )
  const learnedVsInnate = comparisons.find(c =>
    c.baseline.experimentName === '02-innate' &&
    c.treatment.experimentName === '03-learned'
  )

  if (innateVsRandom) {
    lines.push(`   - Innate vs Random: ${innateVsRandom.percentImprovement.toFixed(1)}% difference`)
  }
  if (learnedVsInnate) {
    lines.push(`   - Learned vs Innate: ${learnedVsInnate.percentImprovement.toFixed(1)}% difference`)
    if (learnedVsInnate.percentImprovement > 10) {
      lines.push('   - LEARNING DETECTED: Improvement beyond innate wiring')
    } else {
      lines.push('   - NO LEARNING: Performance not better than innate alone')
    }
  }

  lines.push(
    '',
    '4. COULD RANDOM WEIGHTS DO THIS?',
  )

  const randomResult = results.get('01-random')
  const innateResult = results.get('02-innate')
  if (randomResult && innateResult) {
    lines.push(`   - Random reward: ${randomResult.finalCumulativeReward.toFixed(4)}`)
    lines.push(`   - Innate reward: ${innateResult.finalCumulativeReward.toFixed(4)}`)
    if (innateResult.finalCumulativeReward > randomResult.finalCumulativeReward * 1.1) {
      lines.push('   - Random cannot do this: innate wiring required')
    } else {
      lines.push('   - WARNING: Random performs similarly to innate!')
    }
  }

  lines.push(
    '',
    '5. WOULD THIS SURVIVE SCRUTINY?',
    `   - Conclusion: ${conclusion}`,
    '   - All data preserved for independent verification',
    '',
    '═'.repeat(64),
  )

  return lines.join('\n')
}
