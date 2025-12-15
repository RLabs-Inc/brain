<script lang="ts">
/**
 * World Visualization - Terminal ASCII view of creature navigation
 *
 * Shows:
 * - 2D grid with creature position
 * - Food source location
 * - Chemical gradient (intensity)
 * - Creature heading direction
 * - Neural activity stats
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

// Props
let {
  width = 40,
  height = 20,
  worldWidth = 100,
  worldHeight = 100,
  creatureX = 50,
  creatureY = 50,
  creatureHeading = 0,
  sourceX = 75,
  sourceY = 75,
  forwardOutput = 0,
  turnOutput = 0,
  reward = 0,
  timestep = 0,
  cumulativeReward = 0,
} = $props<{
  width?: number
  height?: number
  worldWidth?: number
  worldHeight?: number
  creatureX?: number
  creatureY?: number
  creatureHeading?: number
  sourceX?: number
  sourceY?: number
  forwardOutput?: number
  turnOutput?: number
  reward?: number
  timestep?: number
  cumulativeReward?: number
}>()

// Scale world coordinates to display coordinates
function toDisplayX(x: number): number {
  return Math.round((x / worldWidth) * (width - 1))
}

function toDisplayY(y: number): number {
  return Math.round((y / worldHeight) * (height - 1))
}

// Get heading character
function getHeadingChar(heading: number): string {
  // Normalize to 0-2π
  const h = ((heading % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)

  // 8 directions
  if (h < Math.PI / 8 || h >= 15 * Math.PI / 8) return '→'
  if (h < 3 * Math.PI / 8) return '↗'
  if (h < 5 * Math.PI / 8) return '↑'
  if (h < 7 * Math.PI / 8) return '↖'
  if (h < 9 * Math.PI / 8) return '←'
  if (h < 11 * Math.PI / 8) return '↙'
  if (h < 13 * Math.PI / 8) return '↓'
  return '↘'
}

// Build the display grid
const grid = $derived.by(() => {
  const lines: string[] = []

  const displayCreatureX = toDisplayX(creatureX)
  const displayCreatureY = toDisplayY(creatureY)
  const displaySourceX = toDisplayX(sourceX)
  const displaySourceY = toDisplayY(sourceY)

  // Header
  lines.push('┌' + '─'.repeat(width) + '┐')

  for (let y = 0; y < height; y++) {
    let line = '│'

    for (let x = 0; x < width; x++) {
      if (x === displayCreatureX && y === displayCreatureY) {
        // Creature with heading
        line += getHeadingChar(creatureHeading)
      } else if (x === displaySourceX && y === displaySourceY) {
        // Food source
        line += '◎'
      } else if (Math.abs(x - displaySourceX) <= 2 && Math.abs(y - displaySourceY) <= 2) {
        // Near food (reward radius)
        line += '·'
      } else {
        line += ' '
      }
    }

    line += '│'
    lines.push(line)
  }

  // Footer
  lines.push('└' + '─'.repeat(width) + '┘')

  return lines.join('\n')
})

// Stats display
const stats = $derived(
  `t=${timestep.toString().padStart(4)} | ` +
  `pos=(${creatureX.toFixed(1)}, ${creatureY.toFixed(1)}) | ` +
  `fwd=${forwardOutput.toFixed(2)} turn=${turnOutput.toFixed(2)} | ` +
  `reward=${cumulativeReward.toFixed(2)}`
)
</script>

<box flexDirection="column">
  <text>{grid}</text>
  <text color="cyan">{stats}</text>
  <text color="gray">◎ = food source | {getHeadingChar(creatureHeading)} = creature (heading: {(creatureHeading * 180 / Math.PI).toFixed(0)}°)</text>
</box>
