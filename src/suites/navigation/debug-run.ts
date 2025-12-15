/**
 * Debug Navigation - See what's actually happening
 */

import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createInnateWorm } from './helpers.ts'
import { core as mx } from '@frost-beta/mlx'

async function debug() {
  console.log('Creating world and creature...')

  const world = createGradient('debug_world', {
    width: 100,
    height: 100,
    sourceX: 75,
    sourceY: 75,
    decayRate: 0.03,
    rewardRadius: 10,
  })

  const creature = createInnateWorm('debug_creature', {
    scale: 0.5,
    learningEnabled: false,
  })

  world.addCreature(creature, { x: 25, y: 25, heading: 0 })

  console.log('Starting position: (25, 25)')
  console.log('Target (food source): (75, 75)')
  console.log('')

  // Run 100 steps and print every 10
  for (let t = 0; t < 100; t++) {
    // Sense
    const stimuli = world.getStimuli(creature)

    // Think
    creature.sense(stimuli)
    await creature.think(1.0)

    // Act
    const actions = creature.act()
    world.applyActions(creature, actions)

    // Get position
    const pos = world.getPosition(creature)

    // Print every 10 steps
    if (t % 10 === 0) {
      const forward = actions.get('forward') ?? 0
      const turn = actions.get('turn') ?? 0
      console.log(`t=${t}: pos=(${pos?.x.toFixed(2)}, ${pos?.y.toFixed(2)}) heading=${(pos?.heading ?? 0).toFixed(2)} | forward=${forward.toFixed(4)} turn=${turn.toFixed(4)}`)
    }

    world.step(1.0)
  }

  const finalPos = world.getPosition(creature)
  console.log('')
  console.log(`Final position: (${finalPos?.x.toFixed(2)}, ${finalPos?.y.toFixed(2)})`)

  // Calculate distance traveled
  const dx = (finalPos?.x ?? 25) - 25
  const dy = (finalPos?.y ?? 25) - 25
  const distanceTraveled = Math.sqrt(dx*dx + dy*dy)
  console.log(`Distance traveled: ${distanceTraveled.toFixed(2)}`)

  creature.destroy()
  world.destroy()
}

debug().catch(console.error)
