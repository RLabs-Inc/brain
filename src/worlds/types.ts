/**
 * World Types - Environment Interface
 *
 * A world is an environment that creatures inhabit.
 * It provides sensory stimuli and receives motor actions.
 * It determines rewards based on creature behavior.
 *
 * NOTE: This file contains TypeScript interfaces (compile-time contracts).
 * Actual reactive state lives in the world implementations (.svelte.ts files).
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import type { Creature } from '../creatures/types.ts'

// ============================================================================
// POSITION AND PHYSICS
// ============================================================================

export interface Position {
  x: number
  y: number
  heading?: number  // Direction in radians (for creatures that can turn)
}

export interface Velocity {
  vx: number
  vy: number
  angular?: number  // Rotational velocity
}

// ============================================================================
// WORLD STATE (for visualization and data collection)
// ============================================================================

export interface CreatureInWorld {
  creature: Creature
  position: Position
  velocity: Velocity
}

export interface WorldState {
  // Identity
  id: string

  // Dimensions
  width: number
  height: number

  // Time
  timestep: number
  elapsedTime: number

  // Creatures in this world
  creatures: Map<string, CreatureInWorld>  // creatureId → state

  // World-specific state (food locations, obstacles, etc.)
  // Implemented by specific world types
  custom: Record<string, unknown>
}

// ============================================================================
// STIMULI AND ACTIONS
// ============================================================================

/**
 * Stimuli are what the creature senses from the world.
 * Map of sensor name → stimulus value (0-1 normalized typically)
 */
export type Stimuli = Map<string, number>

/**
 * Actions are what the creature does in the world.
 * Map of motor name → action value
 */
export type Actions = Map<string, number>

// ============================================================================
// WORLD INTERFACE
// ============================================================================

export interface World {
  // Identity
  readonly id: string

  // Dimensions
  readonly width: number
  readonly height: number

  /**
   * Add a creature to the world at a position.
   */
  addCreature(creature: Creature, position: Position): void

  /**
   * Remove a creature from the world.
   */
  removeCreature(creature: Creature): void

  /**
   * Get what a creature senses at its current position.
   * World → Creature pathway.
   *
   * @param creature - The creature sensing
   * @returns Map of sensor name → stimulus value
   */
  getStimuli(creature: Creature): Stimuli

  /**
   * Apply a creature's actions to the world.
   * Creature → World pathway.
   *
   * @param creature - The creature acting
   * @param actions - Map of motor name → action value
   */
  applyActions(creature: Creature, actions: Actions): void

  /**
   * Advance the world by one timestep.
   * Updates physics, spawns food, etc.
   *
   * @param dt - Timestep in ms
   */
  step(dt: number): void

  /**
   * Get the reward signal for a creature.
   * This is what drives learning.
   *
   * @param creature - The creature to evaluate
   * @returns Reward (positive = good, negative = bad, 0 = neutral)
   */
  getReward(creature: Creature): number

  /**
   * Get current state for visualization/data collection.
   */
  getState(): WorldState

  /**
   * Get a creature's current position.
   */
  getPosition(creature: Creature): Position | undefined

  /**
   * Release all resources.
   */
  destroy(): void
}

// ============================================================================
// WORLD FACTORY
// ============================================================================

export interface WorldOptions {
  // Dimensions
  width?: number
  height?: number

  // Seed for reproducible worlds
  seed?: number

  // World-specific options (food density, obstacles, etc.)
  custom?: Record<string, unknown>
}

/**
 * Factory function signature for worlds.
 */
export type WorldFactory = (id: string, options?: WorldOptions) => World

// ============================================================================
// SPECIFIC WORLD TYPES (for type-safe custom state)
// ============================================================================

/**
 * Gradient world - chemical gradient for chemotaxis experiments
 */
export interface GradientWorldState extends WorldState {
  custom: {
    // Gradient source position
    sourceX: number
    sourceY: number

    // Gradient parameters
    maxConcentration: number
    decayRate: number  // How fast concentration falls off with distance
  }
}

/**
 * Grid world - discrete grid with food and obstacles
 */
export interface GridWorldState extends WorldState {
  custom: {
    // Grid dimensions
    gridWidth: number
    gridHeight: number
    cellSize: number

    // Food locations (grid coordinates)
    foodPositions: Position[]

    // Obstacle locations
    obstaclePositions: Position[]

    // Stats
    totalFoodEaten: number
    totalCollisions: number
  }
}
