/**
 * Gradient World - Chemical Gradient for Chemotaxis Experiments
 *
 * A 2D world with a chemical gradient emanating from a source.
 * Creatures sense the concentration at their location.
 * Reward is based on proximity to the source (finding food).
 *
 * This is the classic C. elegans chemotaxis setup:
 * - Worm starts somewhere in the arena
 * - Food source emits chemical attractant
 * - Worm should learn to follow gradient to food
 *
 * Following sveltui pattern:
 * - $state for reactive properties
 * - Factory function returns World instance
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { SvelteMap } from 'svelte/reactivity'
import type { Creature } from '../creatures/types.ts'
import type {
  World,
  WorldState,
  WorldOptions,
  Position,
  CreatureInWorld,
  Stimuli,
  Actions,
  GradientWorldState
} from './types.ts'

// ============================================================================
// GRADIENT WORLD OPTIONS
// ============================================================================

export interface GradientWorldOptions extends WorldOptions {
  // Source position (default: center)
  sourceX?: number
  sourceY?: number

  // Gradient parameters
  maxConcentration?: number  // Concentration at source (default: 1.0)
  decayRate?: number         // How fast concentration falls off (default: 0.1)

  // Reward parameters
  rewardRadius?: number      // Distance at which max reward is given
  rewardScale?: number       // Multiplier for reward signal

  // Movement parameters
  moveSpeed?: number         // How fast creatures move per action unit
  turnSpeed?: number         // How fast creatures turn per action unit
}

// ============================================================================
// GRADIENT WORLD FACTORY
// ============================================================================

/**
 * Create a gradient world for chemotaxis experiments.
 *
 * The gradient follows: concentration = maxConcentration * exp(-decayRate * distance)
 *
 * @param id - Unique identifier for this world
 * @param options - Configuration options
 */
export function createGradient(id: string, options: GradientWorldOptions = {}): World {
  // World dimensions
  const width = options.width ?? 100
  const height = options.height ?? 100

  // Source position (default: center)
  const sourceX = options.sourceX ?? width / 2
  const sourceY = options.sourceY ?? height / 2

  // Gradient parameters
  const maxConcentration = options.maxConcentration ?? 1.0
  const decayRate = options.decayRate ?? 0.05  // Slower decay for larger arena

  // Reward parameters
  const rewardRadius = options.rewardRadius ?? 10
  const rewardScale = options.rewardScale ?? 1.0

  // Movement parameters
  const moveSpeed = options.moveSpeed ?? 1.0
  const turnSpeed = options.turnSpeed ?? 0.1

  // Reactive state
  let timestep = $state(0)
  let elapsedTime = $state(0)

  // Creatures in this world
  const creatures = $state(new SvelteMap<string, CreatureInWorld>())

  // Track cumulative rewards for analysis
  const cumulativeRewards = $state(new SvelteMap<string, number>())

  // Track previous concentration for temporal comparison (klinokinesis)
  // This is KEY for real chemotaxis - "am I getting closer?"
  const previousConcentration = $state(new SvelteMap<string, number>())

  // ============================================================================
  // HELPER FUNCTIONS
  // ============================================================================

  /**
   * Calculate distance from a position to the source.
   */
  function distanceToSource(x: number, y: number): number {
    const dx = x - sourceX
    const dy = y - sourceY
    return Math.sqrt(dx * dx + dy * dy)
  }

  /**
   * Calculate concentration at a position.
   * Uses exponential decay from source.
   */
  function getConcentration(x: number, y: number): number {
    const distance = distanceToSource(x, y)
    return maxConcentration * Math.exp(-decayRate * distance)
  }

  /**
   * Calculate gradient direction at a position.
   * Returns the direction toward higher concentration.
   */
  function getGradientDirection(x: number, y: number): number {
    const dx = sourceX - x
    const dy = sourceY - y
    return Math.atan2(dy, dx)
  }

  // ============================================================================
  // CREATURE MANAGEMENT
  // ============================================================================

  function addCreature(creature: Creature, position: Position): void {
    creatures.set(creature.id, {
      creature,
      position: {
        x: position.x,
        y: position.y,
        heading: position.heading ?? Math.random() * 2 * Math.PI
      },
      velocity: { vx: 0, vy: 0, angular: 0 }
    })
    cumulativeRewards.set(creature.id, 0)
    // Initialize previous concentration for temporal sensing
    previousConcentration.set(creature.id, getConcentration(position.x, position.y))
  }

  function removeCreature(creature: Creature): void {
    creatures.delete(creature.id)
    cumulativeRewards.delete(creature.id)
    previousConcentration.delete(creature.id)
  }

  function getPosition(creature: Creature): Position | undefined {
    return creatures.get(creature.id)?.position
  }

  // ============================================================================
  // WORLD INTERFACE
  // ============================================================================

  function getStimuli(creature: Creature): Stimuli {
    const state = creatures.get(creature.id)
    if (!state) return new Map()

    const { x, y, heading } = state.position

    // Current concentration at creature's position
    const concentration = getConcentration(x, y)

    // TEMPORAL COMPARISON - The key to real chemotaxis!
    // "Am I getting warmer or colder?"
    const prevConc = previousConcentration.get(creature.id) ?? concentration
    const concentrationDelta = concentration - prevConc  // Positive = getting closer!

    // Update previous concentration for next timestep
    previousConcentration.set(creature.id, concentration)

    // Gradient direction (for sensors that can detect direction)
    const gradientDir = getGradientDirection(x, y)

    // Relative gradient direction (how much to turn to face gradient)
    const currentHeading = heading ?? 0
    let relativeDir = gradientDir - currentHeading

    // Normalize to [-PI, PI]
    while (relativeDir > Math.PI) relativeDir -= 2 * Math.PI
    while (relativeDir < -Math.PI) relativeDir += 2 * Math.PI

    // Left/right sensor asymmetry (worm has bilateral sensors)
    // Positive = more on left, Negative = more on right
    const sensorAsymmetry = Math.sin(relativeDir) * concentration

    // Forward concentration sample (looking ahead)
    const lookAhead = 2.0
    const forwardX = x + lookAhead * Math.cos(currentHeading)
    const forwardY = y + lookAhead * Math.sin(currentHeading)
    const forwardConcentration = getConcentration(forwardX, forwardY)

    // Concentration change in heading direction
    const concentrationChange = forwardConcentration - concentration

    return new Map([
      ['chemical', concentration],                    // Current concentration (0-1)
      ['chemical_left', Math.max(0, sensorAsymmetry)],   // Left sensor
      ['chemical_right', Math.max(0, -sensorAsymmetry)], // Right sensor
      ['gradient_strength', Math.abs(concentrationChange)], // How strong is the gradient here?
      ['improving', concentrationChange > 0 ? 1 : 0],  // Are we heading toward food?

      // NEW: Temporal derivative signals - THE KEY TO REAL CHEMOTAXIS
      // These tell the creature "am I getting warmer or colder?"
      ['concentration_increasing', Math.max(0, concentrationDelta * 10)],  // Getting closer! (amplified)
      ['concentration_decreasing', Math.max(0, -concentrationDelta * 10)], // Getting farther! (amplified)
      ['concentration_delta', concentrationDelta],     // Raw delta (can be negative)
    ])
  }

  function applyActions(creature: Creature, actions: Actions): void {
    const state = creatures.get(creature.id)
    if (!state) return

    // Get movement commands
    const forward = actions.get('forward') ?? 0
    const backward = actions.get('backward') ?? 0
    const turnLeft = actions.get('turn_left') ?? 0
    const turnRight = actions.get('turn_right') ?? 0

    // Net movement
    const netForward = (forward - backward) * moveSpeed
    const netTurn = (turnLeft - turnRight) * turnSpeed

    // Update heading
    state.position.heading = (state.position.heading ?? 0) + netTurn

    // Update position
    const heading = state.position.heading
    state.position.x += netForward * Math.cos(heading)
    state.position.y += netForward * Math.sin(heading)

    // Boundary behavior: wrap around (toroidal world)
    state.position.x = ((state.position.x % width) + width) % width
    state.position.y = ((state.position.y % height) + height) % height
  }

  function step(dt: number): void {
    timestep++
    elapsedTime += dt

    // Could add: source movement, concentration decay, multiple sources, etc.
  }

  function getReward(creature: Creature): number {
    const state = creatures.get(creature.id)
    if (!state) return 0

    const distance = distanceToSource(state.position.x, state.position.y)

    // Reward based on proximity to source
    // Max reward when within rewardRadius, falls off exponentially
    let reward = 0

    if (distance < rewardRadius) {
      // High reward for being at the food
      reward = rewardScale
    } else {
      // Small reward based on concentration (encourages following gradient)
      const concentration = getConcentration(state.position.x, state.position.y)
      reward = concentration * rewardScale * 0.1  // Small shaping reward
    }

    // Track cumulative
    const current = cumulativeRewards.get(creature.id) ?? 0
    cumulativeRewards.set(creature.id, current + reward)

    return reward
  }

  function getState(): GradientWorldState {
    return {
      id,
      width,
      height,
      timestep,
      elapsedTime,
      creatures: new Map(creatures),
      custom: {
        sourceX,
        sourceY,
        maxConcentration,
        decayRate
      }
    }
  }

  function destroy(): void {
    creatures.clear()
    cumulativeRewards.clear()
  }

  // ============================================================================
  // RETURN WORLD OBJECT
  // ============================================================================

  return {
    id,
    width,
    height,
    addCreature,
    removeCreature,
    getStimuli,
    applyActions,
    step,
    getReward,
    getState,
    getPosition,
    destroy
  }
}
