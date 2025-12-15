/**
 * Multi-Episode Learning Test
 *
 * THE REAL TEST OF LEARNING:
 * - Random spawn points for creature AND food each episode
 * - Creature persists across episodes (STDP accumulates)
 * - Compare early episodes vs late episodes
 * - If performance improves, THAT'S REAL LEARNING!
 *
 * Usage: bun run dist/src/suites/navigation/multi-episode.mjs
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'
import { createGradient } from '../../worlds/gradient.svelte.ts'
import { createWorm } from '../../creatures/worm.svelte.ts'

// Configuration
const NUM_EPISODES = 20
const STEPS_PER_EPISODE = 200
const WORLD_SIZE = 100
const SUCCESS_RADIUS = 10

// ANSI colors
const GREEN = '\x1b[32m'
const YELLOW = '\x1b[33m'
const CYAN = '\x1b[36m'
const RED = '\x1b[31m'
const BOLD = '\x1b[1m'
const RESET = '\x1b[0m'

// Random number generator with seed
function createRNG(seed: number) {
  let state = seed
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff
    return state / 0x7fffffff
  }
}

// Episode result
interface EpisodeResult {
  episode: number
  startX: number
  startY: number
  foodX: number
  foodY: number
  endX: number
  endY: number
  startDistance: number
  endDistance: number
  reachedFood: boolean
  totalReward: number
  stepsToFood: number | null
}

async function runEpisode(
  creature: ReturnType<typeof createWorm>,
  foodX: number,
  foodY: number,
  startX: number,
  startY: number,
  episodeNum: number
): Promise<EpisodeResult> {
  // Create a fresh world for this episode with the food at random location
  const world = createGradient(`episode_${episodeNum}_world`, {
    width: WORLD_SIZE,
    height: WORLD_SIZE,
    sourceX: foodX,
    sourceY: foodY,
    decayRate: 0.03,
    rewardRadius: SUCCESS_RADIUS,
  })

  // Add creature at random start position
  world.addCreature(creature, { x: startX, y: startY, heading: Math.random() * 2 * Math.PI })

  const startDistance = Math.sqrt(Math.pow(startX - foodX, 2) + Math.pow(startY - foodY, 2))
  let totalReward = 0
  let reachedFood = false
  let stepsToFood: number | null = null

  // Run episode
  for (let step = 0; step < STEPS_PER_EPISODE; step++) {
    // Sense
    const stimuli = world.getStimuli(creature)
    creature.sense(stimuli)

    // Think
    await creature.think(1.0)

    // Act
    const actions = creature.act()
    world.applyActions(creature, actions)

    // Get reward
    const reward = world.getReward(creature)
    creature.setReward(reward)
    totalReward += reward

    // World step
    world.step(1.0)

    // Check if reached food
    const pos = world.getPosition(creature)
    if (pos) {
      const distToFood = Math.sqrt(Math.pow(pos.x - foodX, 2) + Math.pow(pos.y - foodY, 2))
      if (distToFood < SUCCESS_RADIUS && !reachedFood) {
        reachedFood = true
        stepsToFood = step + 1
      }
    }
  }

  // Get final position
  const finalPos = world.getPosition(creature)
  const endX = finalPos?.x ?? startX
  const endY = finalPos?.y ?? startY
  const endDistance = Math.sqrt(Math.pow(endX - foodX, 2) + Math.pow(endY - foodY, 2))

  // IMPORTANT: Remove creature from world but DON'T destroy it
  // The creature persists with its learned weights!
  world.removeCreature(creature)
  world.destroy()

  return {
    episode: episodeNum,
    startX,
    startY,
    foodX,
    foodY,
    endX,
    endY,
    startDistance,
    endDistance,
    reachedFood,
    totalReward,
    stepsToFood,
  }
}

async function main() {
  console.log(`${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════╗${RESET}`)
  console.log(`${BOLD}${GREEN}║       MULTI-EPISODE LEARNING TEST - The Real Test!            ║${RESET}`)
  console.log(`${BOLD}${GREEN}╚════════════════════════════════════════════════════════════════╝${RESET}`)
  console.log('')
  console.log(`${CYAN}HYPOTHESIS:${RESET} A creature with STDP learning will improve its`)
  console.log(`            navigation performance across multiple episodes.`)
  console.log('')
  console.log(`${YELLOW}Setup:${RESET}`)
  console.log(`  - ${NUM_EPISODES} episodes`)
  console.log(`  - ${STEPS_PER_EPISODE} steps per episode`)
  console.log(`  - Random spawn points for creature AND food each episode`)
  console.log(`  - Same creature persists (STDP accumulates learning)`)
  console.log('')

  // Create creature ONCE - it persists across all episodes
  const creature = createWorm('learning_creature', {
    scale: 0.5,
    learningEnabled: true,  // STDP enabled!
  })

  // RNG for reproducible random spawns
  const rng = createRNG(42)

  const results: EpisodeResult[] = []

  console.log(`${BOLD}Running episodes...${RESET}`)
  console.log('')

  for (let ep = 1; ep <= NUM_EPISODES; ep++) {
    // Random positions (with margin from edges)
    const margin = 15
    const foodX = margin + rng() * (WORLD_SIZE - 2 * margin)
    const foodY = margin + rng() * (WORLD_SIZE - 2 * margin)
    const startX = margin + rng() * (WORLD_SIZE - 2 * margin)
    const startY = margin + rng() * (WORLD_SIZE - 2 * margin)

    const result = await runEpisode(creature, foodX, foodY, startX, startY, ep)
    results.push(result)

    // Force garbage collection between episodes to free Metal resources
    // @ts-ignore - Bun.gc() may not be in type definitions
    if (typeof Bun !== 'undefined' && Bun.gc) {
      Bun.gc(true)  // true = aggressive GC
    }

    // Print progress
    const status = result.reachedFood
      ? `${GREEN}FOUND FOOD in ${result.stepsToFood} steps!${RESET}`
      : `${YELLOW}End dist: ${result.endDistance.toFixed(1)}${RESET}`

    const improvement = result.startDistance - result.endDistance
    const impStr = improvement > 0
      ? `${GREEN}+${improvement.toFixed(1)}${RESET}`
      : `${RED}${improvement.toFixed(1)}${RESET}`

    console.log(`  Episode ${ep.toString().padStart(2)}: Start dist: ${result.startDistance.toFixed(1).padStart(5)} → ${status} (${impStr})`)
  }

  // Analyze results
  console.log('')
  console.log(`${BOLD}═══════════════════════════════════════════════════════════════${RESET}`)
  console.log(`${BOLD}                        ANALYSIS${RESET}`)
  console.log(`${BOLD}═══════════════════════════════════════════════════════════════${RESET}`)
  console.log('')

  // Split into early and late episodes
  const earlyEpisodes = results.slice(0, Math.floor(NUM_EPISODES / 2))
  const lateEpisodes = results.slice(Math.floor(NUM_EPISODES / 2))

  // Metrics
  const earlySuccesses = earlyEpisodes.filter(r => r.reachedFood).length
  const lateSuccesses = lateEpisodes.filter(r => r.reachedFood).length

  const earlyAvgReward = earlyEpisodes.reduce((s, r) => s + r.totalReward, 0) / earlyEpisodes.length
  const lateAvgReward = lateEpisodes.reduce((s, r) => s + r.totalReward, 0) / lateEpisodes.length

  const earlyAvgImprovement = earlyEpisodes.reduce((s, r) => s + (r.startDistance - r.endDistance), 0) / earlyEpisodes.length
  const lateAvgImprovement = lateEpisodes.reduce((s, r) => s + (r.startDistance - r.endDistance), 0) / lateEpisodes.length

  console.log(`${CYAN}Early Episodes (1-${Math.floor(NUM_EPISODES / 2)}):${RESET}`)
  console.log(`  Success rate: ${earlySuccesses}/${earlyEpisodes.length} (${(earlySuccesses / earlyEpisodes.length * 100).toFixed(1)}%)`)
  console.log(`  Avg reward: ${earlyAvgReward.toFixed(2)}`)
  console.log(`  Avg distance improvement: ${earlyAvgImprovement.toFixed(1)}`)
  console.log('')

  console.log(`${CYAN}Late Episodes (${Math.floor(NUM_EPISODES / 2) + 1}-${NUM_EPISODES}):${RESET}`)
  console.log(`  Success rate: ${lateSuccesses}/${lateEpisodes.length} (${(lateSuccesses / lateEpisodes.length * 100).toFixed(1)}%)`)
  console.log(`  Avg reward: ${lateAvgReward.toFixed(2)}`)
  console.log(`  Avg distance improvement: ${lateAvgImprovement.toFixed(1)}`)
  console.log('')

  // Learning detection
  console.log(`${BOLD}═══════════════════════════════════════════════════════════════${RESET}`)
  console.log(`${BOLD}                      THE VERDICT${RESET}`)
  console.log(`${BOLD}═══════════════════════════════════════════════════════════════${RESET}`)
  console.log('')

  const rewardImprovement = ((lateAvgReward - earlyAvgReward) / earlyAvgReward * 100)
  const successImprovement = lateSuccesses - earlySuccesses
  const distImprovement = lateAvgImprovement - earlyAvgImprovement

  const learningDetected = rewardImprovement > 10 || successImprovement > 1 || distImprovement > 5

  if (learningDetected) {
    console.log(`${BOLD}${GREEN}🎉 LEARNING DETECTED!${RESET}`)
    console.log('')
    console.log(`Evidence:`)
    if (rewardImprovement > 10) {
      console.log(`  - Reward improved by ${rewardImprovement.toFixed(1)}%`)
    }
    if (successImprovement > 0) {
      console.log(`  - ${successImprovement} more successes in late episodes`)
    }
    if (distImprovement > 5) {
      console.log(`  - Distance improvement increased by ${distImprovement.toFixed(1)}`)
    }
  } else {
    console.log(`${BOLD}${RED}❌ NO LEARNING DETECTED${RESET}`)
    console.log('')
    console.log(`The creature did not show improvement across episodes.`)
    console.log(`This could mean:`)
    console.log(`  - STDP learning rate is too low`)
    console.log(`  - The reward signal isn't strong enough`)
    console.log(`  - The task is too hard for the current architecture`)
    console.log(`  - Or... learning simply doesn't work yet`)
  }

  console.log('')
  console.log(`${BOLD}THE SLAP:${RESET}`)
  console.log(`  - Did we cherry-pick these results? NO - all episodes shown`)
  console.log(`  - Could this be random variation? Check the numbers above`)
  console.log(`  - Is this real learning or statistics? YOU decide`)
  console.log('')

  // Cleanup
  creature.destroy()
}

main().catch(console.error)
