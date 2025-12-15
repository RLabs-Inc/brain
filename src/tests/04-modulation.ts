/**
 * Test 04: Neuromodulation
 *
 * Tests: Do neuromodulators affect plasticity?
 *
 * What we test:
 * 1. Modulators start at baseline
 * 2. Signals (reward/punishment) change modulator levels
 * 3. Modulators decay toward baseline
 * 4. Plasticity gate combines all modulators
 * 5. High serotonin reduces plasticity (satiation)
 *
 * The Slap: Modulation must have specific, testable effects.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import {
  startSuite,
  endSuite,
  test,
  assertEqual,
  assertTrue,
  assertGreater,
  assertLess,
  assertInRange,
  getScalar,
  withSlap,
} from './utils.ts'
import {
  allocateModulation,
  releaseModulation,
  getModulatorLevels,
  getPlasticityGateById,
  signalRewardById,
  signalPunishmentById,
  signalNoveltyById,
  signalSatiationById,
  decayModulatorsById,
  resetModulators,
  isModulationAllocated,
  ModulatorDefaults,
} from '../core/modulation.svelte.ts'

export function runModulationTests() {
  startSuite('04 - Neuromodulation')

  // -------------------------------------------------------------------------
  // Test 1: Modulation allocation works
  // -------------------------------------------------------------------------
  test('Modulation system allocates correctly', () => {
    const index = allocateModulation('mod_test1')
    const exists = isModulationAllocated('mod_test1')

    releaseModulation('mod_test1')
    const existsAfter = isModulationAllocated('mod_test1')

    return withSlap(
      assertTrue(
        exists && !existsAfter,
        `Allocated: ${exists}, After release: ${existsAfter}`
      ),
      'Registry tracks allocation state correctly'
    )
  })

  // -------------------------------------------------------------------------
  // Test 2: Dopamine starts at baseline
  // -------------------------------------------------------------------------
  test('Dopamine starts at baseline (0)', () => {
    allocateModulation('mod_test2')

    const levels = getModulatorLevels('mod_test2')
    mx.eval(levels.dopamine)
    const da = levels.dopamine.item() as number

    releaseModulation('mod_test2')

    return withSlap(
      assertEqual(da, ModulatorDefaults.dopamine.baseline, `Dopamine: ${da}`),
      'Baseline proves initialization, not random'
    )
  })

  // -------------------------------------------------------------------------
  // Test 3: Serotonin starts at baseline
  // -------------------------------------------------------------------------
  test('Serotonin starts at baseline (0.5)', () => {
    allocateModulation('mod_test3')

    const levels = getModulatorLevels('mod_test3')
    mx.eval(levels.serotonin)
    const ser = levels.serotonin.item() as number

    releaseModulation('mod_test3')

    return withSlap(
      assertInRange(ser, 0.49, 0.51, `Serotonin: ${ser}`),
      'Tonic serotonin level for mood baseline'
    )
  })

  // -------------------------------------------------------------------------
  // Test 4: Reward increases dopamine
  // -------------------------------------------------------------------------
  test('Reward signal increases dopamine', () => {
    allocateModulation('mod_test4')

    const levelsBefore = getModulatorLevels('mod_test4')
    mx.eval(levelsBefore.dopamine)
    const daBefore = levelsBefore.dopamine.item() as number

    signalRewardById('mod_test4', 0.5)

    const levelsAfter = getModulatorLevels('mod_test4')
    mx.eval(levelsAfter.dopamine)
    const daAfter = levelsAfter.dopamine.item() as number

    releaseModulation('mod_test4')

    return withSlap(
      assertGreater(daAfter, daBefore, `DA: ${daBefore} → ${daAfter}`),
      'Reward → dopamine proves reward prediction error mechanism'
    )
  })

  // -------------------------------------------------------------------------
  // Test 5: Punishment decreases dopamine
  // -------------------------------------------------------------------------
  test('Punishment decreases dopamine', () => {
    allocateModulation('mod_test5')

    // Start with some dopamine
    signalRewardById('mod_test5', 0.5)
    const levelsBefore = getModulatorLevels('mod_test5')
    mx.eval(levelsBefore.dopamine)
    const daBefore = levelsBefore.dopamine.item() as number

    signalPunishmentById('mod_test5', 0.8)

    const levelsAfter = getModulatorLevels('mod_test5')
    mx.eval(levelsAfter.dopamine)
    const daAfter = levelsAfter.dopamine.item() as number

    releaseModulation('mod_test5')

    return withSlap(
      assertLess(daAfter, daBefore, `DA: ${daBefore} → ${daAfter}`),
      'Punishment reduces DA - negative prediction error'
    )
  })

  // -------------------------------------------------------------------------
  // Test 6: Novelty increases norepinephrine
  // -------------------------------------------------------------------------
  test('Novelty signal increases norepinephrine', () => {
    allocateModulation('mod_test6')

    const levelsBefore = getModulatorLevels('mod_test6')
    mx.eval(levelsBefore.norepinephrine)
    const neBefore = levelsBefore.norepinephrine.item() as number

    signalNoveltyById('mod_test6', 0.5)

    const levelsAfter = getModulatorLevels('mod_test6')
    mx.eval(levelsAfter.norepinephrine)
    const neAfter = levelsAfter.norepinephrine.item() as number

    releaseModulation('mod_test6')

    return withSlap(
      assertGreater(neAfter, neBefore, `NE: ${neBefore} → ${neAfter}`),
      'Novelty → NE proves attention/arousal mechanism'
    )
  })

  // -------------------------------------------------------------------------
  // Test 7: Satiation increases serotonin
  // -------------------------------------------------------------------------
  test('Satiation signal increases serotonin', () => {
    allocateModulation('mod_test7')

    const levelsBefore = getModulatorLevels('mod_test7')
    mx.eval(levelsBefore.serotonin)
    const serBefore = levelsBefore.serotonin.item() as number

    signalSatiationById('mod_test7', 0.3)

    const levelsAfter = getModulatorLevels('mod_test7')
    mx.eval(levelsAfter.serotonin)
    const serAfter = levelsAfter.serotonin.item() as number

    releaseModulation('mod_test7')

    return withSlap(
      assertGreater(serAfter, serBefore, `5-HT: ${serBefore} → ${serAfter}`),
      'Satiation → 5-HT proves fullness/satisfaction mechanism'
    )
  })

  // -------------------------------------------------------------------------
  // Test 8: Modulators decay toward baseline
  // -------------------------------------------------------------------------
  test('Modulators decay toward baseline', () => {
    allocateModulation('mod_test8', { dopamineDecay: 0.5 }) // Fast decay for test

    // Spike dopamine
    signalRewardById('mod_test8', 1.0)

    const levelsBefore = getModulatorLevels('mod_test8')
    mx.eval(levelsBefore.dopamine)
    const daBefore = levelsBefore.dopamine.item() as number

    // Decay several times
    for (let i = 0; i < 10; i++) {
      decayModulatorsById('mod_test8')
    }

    const levelsAfter = getModulatorLevels('mod_test8')
    mx.eval(levelsAfter.dopamine)
    const daAfter = levelsAfter.dopamine.item() as number

    releaseModulation('mod_test8')

    return withSlap(
      assertLess(
        Math.abs(daAfter),
        Math.abs(daBefore),
        `DA: ${daBefore.toFixed(3)} → ${daAfter.toFixed(3)}`
      ),
      'Decay returns to baseline - transient signals'
    )
  })

  // -------------------------------------------------------------------------
  // Test 9: Plasticity gate responds to reward
  // -------------------------------------------------------------------------
  test('Plasticity gate increases with reward', () => {
    allocateModulation('mod_test9')

    mx.eval(getPlasticityGateById('mod_test9'))
    const gateBefore = getPlasticityGateById('mod_test9').item() as number

    signalRewardById('mod_test9', 0.5)

    mx.eval(getPlasticityGateById('mod_test9'))
    const gateAfter = getPlasticityGateById('mod_test9').item() as number

    releaseModulation('mod_test9')

    return withSlap(
      assertGreater(
        gateAfter,
        gateBefore,
        `Gate: ${gateBefore.toFixed(3)} → ${gateAfter.toFixed(3)}`
      ),
      'Reward opens plasticity gate - enables learning'
    )
  })

  // -------------------------------------------------------------------------
  // Test 10: High serotonin reduces plasticity gate
  // -------------------------------------------------------------------------
  test('High serotonin reduces plasticity (satiation effect)', () => {
    allocateModulation('mod_test10')

    mx.eval(getPlasticityGateById('mod_test10'))
    const gateBefore = getPlasticityGateById('mod_test10').item() as number

    // Heavy satiation
    signalSatiationById('mod_test10', 0.5)
    signalSatiationById('mod_test10', 0.5)

    mx.eval(getPlasticityGateById('mod_test10'))
    const gateAfter = getPlasticityGateById('mod_test10').item() as number

    releaseModulation('mod_test10')

    return withSlap(
      assertLess(
        gateAfter,
        gateBefore,
        `Gate: ${gateBefore.toFixed(3)} → ${gateAfter.toFixed(3)}`
      ),
      'Satiation closes plasticity - no learning when satisfied'
    )
  })

  // -------------------------------------------------------------------------
  // Test 11: Reset restores baseline
  // -------------------------------------------------------------------------
  test('Reset restores all modulators to baseline', () => {
    const index = allocateModulation('mod_test11')

    // Change all modulators
    signalRewardById('mod_test11', 1.0)
    signalNoveltyById('mod_test11', 1.0)
    signalSatiationById('mod_test11', 1.0)

    // Reset using the actual allocated index
    resetModulators(index)

    const levelsAfter = getModulatorLevels('mod_test11')
    mx.eval(
      levelsAfter.dopamine,
      levelsAfter.serotonin,
      levelsAfter.norepinephrine
    )

    const daAfter = levelsAfter.dopamine.item() as number
    const serAfter = levelsAfter.serotonin.item() as number
    const neAfter = levelsAfter.norepinephrine.item() as number

    releaseModulation('mod_test11')

    const atBaseline =
      Math.abs(daAfter) < 0.1 &&
      Math.abs(serAfter - 0.5) < 0.1 &&
      Math.abs(neAfter - 0.3) < 0.1

    return withSlap(
      assertTrue(
        atBaseline,
        `DA: ${daAfter.toFixed(2)}, 5-HT: ${serAfter.toFixed(
          2
        )}, NE: ${neAfter.toFixed(2)}`
      ),
      'Reset enables clean slate for new episode'
    )
  })

  endSuite()
}

// Run if executed directly
if (import.meta) {
  runModulationTests()
}
