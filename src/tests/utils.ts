/**
 * Test Utilities - Assertions and Reporting
 *
 * Simple test framework for validating brain primitives.
 * Following the Honesty Protocol: no fake successes.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'

// ============================================================================
// TEST RESULT TRACKING
// ============================================================================

export interface TestResult {
  name: string
  passed: boolean
  message: string
  actual?: unknown
  expected?: unknown
  slap?: string  // Honesty check result
}

export interface TestSuite {
  name: string
  results: TestResult[]
  startTime: number
  endTime?: number
}

let currentSuite: TestSuite | null = null
const allSuites: TestSuite[] = []

// ============================================================================
// SUITE MANAGEMENT
// ============================================================================

export function startSuite(name: string) {
  currentSuite = {
    name,
    results: [],
    startTime: Date.now(),
  }
  console.log()
  console.log('='.repeat(60))
  console.log(`TEST SUITE: ${name}`)
  console.log('='.repeat(60))
}

export function endSuite() {
  if (!currentSuite) return

  currentSuite.endTime = Date.now()
  allSuites.push(currentSuite)

  const passed = currentSuite.results.filter(r => r.passed).length
  const total = currentSuite.results.length
  const duration = currentSuite.endTime - currentSuite.startTime

  console.log('-'.repeat(60))
  console.log(`Results: ${passed}/${total} passed (${duration}ms)`)
  console.log()

  currentSuite = null
}

// ============================================================================
// ASSERTIONS
// ============================================================================

export function test(name: string, fn: () => TestResult | void) {
  process.stdout.write(`  ${name}... `)

  try {
    const result = fn()

    if (result) {
      currentSuite?.results.push(result)
      if (result.passed) {
        console.log('\x1b[32mPASS\x1b[0m')
        if (result.message) {
          console.log(`    ${result.message}`)
        }
      } else {
        console.log('\x1b[31mFAIL\x1b[0m')
        console.log(`    ${result.message}`)
        if (result.expected !== undefined) {
          console.log(`    Expected: ${result.expected}`)
          console.log(`    Actual: ${result.actual}`)
        }
      }
      if (result.slap) {
        console.log(`    \x1b[33m[SLAP] ${result.slap}\x1b[0m`)
      }
    } else {
      // No explicit result means pass
      currentSuite?.results.push({ name, passed: true, message: '' })
      console.log('\x1b[32mPASS\x1b[0m')
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    currentSuite?.results.push({
      name,
      passed: false,
      message: `Exception: ${message}`,
    })
    console.log('\x1b[31mFAIL\x1b[0m')
    console.log(`    Exception: ${message}`)
  }
}

/**
 * Assert equality with tolerance for numbers.
 */
export function assertEqual(
  actual: unknown,
  expected: unknown,
  message: string = '',
  tolerance: number = 1e-6
): TestResult {
  let passed = false

  if (typeof actual === 'number' && typeof expected === 'number') {
    passed = Math.abs(actual - expected) < tolerance
  } else {
    passed = actual === expected
  }

  return {
    name: message,
    passed,
    message: passed ? message : `Expected ${expected}, got ${actual}`,
    actual,
    expected,
  }
}

/**
 * Assert a value is true.
 */
export function assertTrue(value: boolean, message: string = ''): TestResult {
  return {
    name: message,
    passed: value,
    message: value ? message : `Expected true, got false`,
    actual: value,
    expected: true,
  }
}

/**
 * Assert a value is false.
 */
export function assertFalse(value: boolean, message: string = ''): TestResult {
  return {
    name: message,
    passed: !value,
    message: !value ? message : `Expected false, got true`,
    actual: value,
    expected: false,
  }
}

/**
 * Assert value is greater than threshold.
 */
export function assertGreater(
  actual: number,
  threshold: number,
  message: string = ''
): TestResult {
  const passed = actual > threshold
  return {
    name: message,
    passed,
    message: passed ? `${actual} > ${threshold}` : `Expected > ${threshold}, got ${actual}`,
    actual,
    expected: `> ${threshold}`,
  }
}

/**
 * Assert value is less than threshold.
 */
export function assertLess(
  actual: number,
  threshold: number,
  message: string = ''
): TestResult {
  const passed = actual < threshold
  return {
    name: message,
    passed,
    message: passed ? `${actual} < ${threshold}` : `Expected < ${threshold}, got ${actual}`,
    actual,
    expected: `< ${threshold}`,
  }
}

/**
 * Assert value is within range.
 */
export function assertInRange(
  actual: number,
  min: number,
  max: number,
  message: string = ''
): TestResult {
  const passed = actual >= min && actual <= max
  return {
    name: message,
    passed,
    message: passed
      ? `${actual} in [${min}, ${max}]`
      : `Expected in [${min}, ${max}], got ${actual}`,
    actual,
    expected: `[${min}, ${max}]`,
  }
}

/**
 * Assert array has expected length.
 */
export function assertLength(
  arr: unknown[],
  expected: number,
  message: string = ''
): TestResult {
  const passed = arr.length === expected
  return {
    name: message,
    passed,
    message: passed ? `Length is ${expected}` : `Expected length ${expected}, got ${arr.length}`,
    actual: arr.length,
    expected,
  }
}

// ============================================================================
// MLX HELPERS
// ============================================================================

/**
 * Get scalar value from MLX array.
 */
export function getScalar(arr: ReturnType<typeof mx.array>): number {
  mx.eval(arr)
  return arr.item() as number
}

/**
 * Get array values from MLX array.
 */
export function getArray(arr: ReturnType<typeof mx.array>): number[] {
  mx.eval(arr)
  return arr.tolist() as number[]
}

/**
 * Get boolean array values.
 */
export function getBoolArray(arr: ReturnType<typeof mx.array>): boolean[] {
  mx.eval(arr)
  return arr.tolist() as boolean[]
}

/**
 * Count true values in boolean array.
 */
export function countTrue(arr: ReturnType<typeof mx.array>): number {
  mx.eval(arr)
  return getScalar(mx.sum(arr))
}

// ============================================================================
// HONESTY PROTOCOL - "THE SLAP"
// ============================================================================

/**
 * Add a "slap" check to a test result.
 * This documents WHY this result proves something real.
 */
export function withSlap(result: TestResult, slapCheck: string): TestResult {
  return { ...result, slap: slapCheck }
}

/**
 * Check if result could happen by chance.
 * Returns a warning string if suspicious.
 */
export function checkByChance(
  actual: number,
  baseline: number,
  description: string
): string | undefined {
  // If actual is very close to baseline, it might be chance
  const ratio = Math.abs(actual - baseline) / Math.max(Math.abs(baseline), 1)
  if (ratio < 0.1) {
    return `WARNING: ${description} - result (${actual}) is very close to baseline (${baseline}), might be chance`
  }
  return undefined
}

// ============================================================================
// FINAL REPORT
// ============================================================================

export function printFinalReport() {
  console.log()
  console.log('='.repeat(60))
  console.log('FINAL REPORT')
  console.log('='.repeat(60))

  let totalPassed = 0
  let totalTests = 0

  for (const suite of allSuites) {
    const passed = suite.results.filter(r => r.passed).length
    const total = suite.results.length
    const status = passed === total ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m'

    console.log(`  ${suite.name}: ${passed}/${total} ${status}`)
    totalPassed += passed
    totalTests += total
  }

  console.log('-'.repeat(60))
  const allPassed = totalPassed === totalTests
  const finalStatus = allPassed ? '\x1b[32mALL TESTS PASSED\x1b[0m' : '\x1b[31mSOME TESTS FAILED\x1b[0m'
  console.log(`  Total: ${totalPassed}/${totalTests} ${finalStatus}`)
  console.log()

  // Return exit code
  return allPassed ? 0 : 1
}

/**
 * Reset all test state (useful between test runs).
 */
export function resetTests() {
  currentSuite = null
  allSuites.length = 0
}

/**
 * Get total passed/failed counts.
 */
export function getTotalResults(): { passed: number; failed: number } {
  let passed = 0
  let failed = 0

  for (const suite of allSuites) {
    for (const result of suite.results) {
      if (result.passed) {
        passed++
      } else {
        failed++
      }
    }
  }

  return { passed, failed }
}
