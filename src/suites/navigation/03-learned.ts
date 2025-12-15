/**
 * Experiment 03 - Learning Enabled
 *
 * HYPOTHESIS: STDP learning can improve navigation beyond innate wiring.
 * EXPECTED OUTCOME: UNKNOWN - This is the real test!
 *
 * This is what we're really testing. If learning helps:
 * - STDP + reward discovered something innate wiring didn't encode
 * - The creature learned to follow gradients, not just local search
 *
 * If learning doesn't help:
 * - Maybe innate wiring is already optimal for this task
 * - Maybe our STDP parameters need tuning
 * - Maybe the task doesn't require learning
 *
 * THE SLAP: We don't know what to expect. That's the point.
 * This is a real experiment, not a demonstration.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Experiment, ExperimentResult, Analysis } from '../types.ts'
import { runExperiment, analyzeResult } from '../runner.svelte.ts'
import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createInnateWorm } from './helpers.ts'

// ============================================================================
// EXPERIMENT DEFINITION
// ============================================================================

export function createLearnedExperiment(): Experiment {
  let creature: ReturnType<typeof createInnateWorm> | null = null
  let world: ReturnType<typeof createGradient> | null = null

  return {
    name: '03-learned',
    description: 'Innate wiring WITH STDP learning enabled',
    hypothesis: 'Learning can improve navigation beyond innate wiring alone',
    expectedOutcome: 'unknown',  // THE REAL TEST
    slapQuestions: [
      'Did weights actually change during the experiment?',
      'Is improvement due to learning or random fluctuation?',
      'Would multiple runs show consistent improvement?',
      'Is the improvement biologically meaningful or just statistical noise?',
    ],

    setup() {
      world = createGradient('learning_world', {
        width: 100,
        height: 100,
        sourceX: 75,
        sourceY: 75,
        decayRate: 0.03,
        rewardRadius: 10,
      })

      // Create creature with INNATE wiring AND learning ENABLED
      creature = createInnateWorm('learning_creature', {
        scale: 0.5,
        learningEnabled: true,  // STDP enabled!
      })

      world.addCreature(creature, { x: 25, y: 25, heading: 0 })

      return { creature, world }
    },

    async run(timesteps: number, onStep?: any): Promise<ExperimentResult> {
      if (!creature || !world) {
        throw new Error('Must call setup() before run()')
      }

      return runExperiment(creature, world, timesteps, { dt: 1.0 }, onStep)
    },

    analyze(result: ExperimentResult): Analysis {
      const base = analyzeResult(result)

      // This is the key experiment - we genuinely don't know the outcome
      base.custom.isRealTest = true
      base.custom.learningWasEnabled = true

      // Check for learning signatures
      // 1. Did reward improve over time? (first half vs second half)
      const data = result.data
      const midpoint = Math.floor(data.length / 2)
      const firstHalf = data.slice(0, midpoint)
      const secondHalf = data.slice(midpoint)

      const firstHalfReward = firstHalf.reduce((sum, d) => sum + d.reward, 0)
      const secondHalfReward = secondHalf.reduce((sum, d) => sum + d.reward, 0)

      const improvementRatio = secondHalfReward / (firstHalfReward + 0.001)
      base.custom.firstHalfReward = firstHalfReward
      base.custom.secondHalfReward = secondHalfReward
      base.custom.improvementRatio = improvementRatio

      // Learning signature: second half notably better than first
      const showsLearningCurve = improvementRatio > 1.2

      base.slapAnswers.set(
        'Did weights actually change?',
        'Check: Compare initial vs final weights (not tracked in this version)'
      )
      base.slapAnswers.set(
        'Is improvement due to learning?',
        showsLearningCurve
          ? `Possible - second half ${((improvementRatio - 1) * 100).toFixed(1)}% better than first`
          : `Unclear - no learning curve detected (ratio: ${improvementRatio.toFixed(2)})`
      )
      base.slapAnswers.set(
        'Would multiple runs show consistency?',
        'TODO: Run multiple trials with different seeds'
      )
      base.slapAnswers.set(
        'Is improvement biologically meaningful?',
        result.finalCumulativeReward > 1.0
          ? 'Possibly - creature accumulated significant reward'
          : 'Unclear - total reward was low'
      )

      return base
    },

    teardown() {
      creature?.destroy()
      world?.destroy()
      creature = null
      world = null
    },
  }
}
