/**
 * Void World - Empty Environment for Isolated Testing
 *
 * The void is an empty world with no stimuli, no physics, no rewards.
 * Creatures exist but have nothing to interact with.
 *
 * Use this for:
 * - Testing creature mechanics in isolation
 * - Verifying brain activity without world interference
 * - Baseline "no environment" experiments
 *
 * Following sveltui pattern:
 * - $state for reactive properties that UI might observe
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
  Velocity,
  CreatureInWorld,
  Stimuli,
  Actions
} from './types.ts'

// ============================================================================
// VOID WORLD FACTORY
// ============================================================================

export interface VoidWorldOptions extends WorldOptions {
  // No additional options for void world
}

/**
 * Create a void world - empty space for isolated testing.
 *
 * @param id - Unique identifier for this world
 * @param options - Configuration options
 */
export function createVoid(id: string, options: VoidWorldOptions = {}): World {
  // World dimensions (default to unit square)
  const width = options.width ?? 100
  const height = options.height ?? 100

  // Reactive state for UI observation
  let timestep = $state(0)
  let elapsedTime = $state(0)

  // Creatures in this world
  const creatures = $state(new SvelteMap<string, CreatureInWorld>())

  // ============================================================================
  // CREATURE MANAGEMENT
  // ============================================================================

  function addCreature(creature: Creature, position: Position): void {
    creatures.set(creature.id, {
      creature,
      position: { ...position },
      velocity: { vx: 0, vy: 0, angular: 0 }
    })
  }

  function removeCreature(creature: Creature): void {
    creatures.delete(creature.id)
  }

  function getPosition(creature: Creature): Position | undefined {
    return creatures.get(creature.id)?.position
  }

  // ============================================================================
  // WORLD INTERFACE
  // ============================================================================

  function getStimuli(_creature: Creature): Stimuli {
    // Void world provides no stimuli
    return new Map()
  }

  function applyActions(creature: Creature, actions: Actions): void {
    // Void world ignores actions - nothing to affect
    // But we could optionally update position based on locomotion
    const state = creatures.get(creature.id)
    if (!state) return

    // If creature has movement motors, update position (optional behavior)
    const forward = actions.get('forward') ?? 0
    const turn = actions.get('turn') ?? 0

    if (forward !== 0 || turn !== 0) {
      // Update heading
      state.position.heading = (state.position.heading ?? 0) + turn * 0.1

      // Update position
      const heading = state.position.heading ?? 0
      state.position.x += forward * Math.cos(heading)
      state.position.y += forward * Math.sin(heading)

      // Wrap around edges
      state.position.x = ((state.position.x % width) + width) % width
      state.position.y = ((state.position.y % height) + height) % height
    }
  }

  function step(dt: number): void {
    // Void world has no physics to update
    timestep++
    elapsedTime += dt
  }

  function getReward(_creature: Creature): number {
    // Void world provides no reward signal
    return 0
  }

  function getState(): WorldState {
    return {
      id,
      width,
      height,
      timestep,
      elapsedTime,
      creatures: new Map(creatures),  // Copy for immutability
      custom: {}
    }
  }

  function destroy(): void {
    creatures.clear()
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
