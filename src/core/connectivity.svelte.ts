/**
 * Connectivity Patterns - Reusable Wiring Templates
 *
 * Following sveltui pattern EXACTLY:
 * - ALL computation on GPU - NEVER convert to JS
 * - Direct GPU array operations
 *
 * These are connectivity patterns that evolution discovered:
 * - Lateral inhibition (contrast enhancement)
 * - Center-surround (edge detection)
 * - Topographic mapping (spatial organization)
 * - Feedforward/feedback (hierarchical processing)
 * - Recurrent (working memory, attractor dynamics)
 * - Winner-take-all (decision making)
 *
 * All functions return GPU arrays ready for allocateSynapseGroup.
 *
 * @author Rodrigo Luglio & Claude
 * @date December 2025
 */

import { core as mx } from '@frost-beta/mlx'

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Connectivity result - ready for allocateSynapseGroup.
 */
export interface Connectivity {
  preIndices: ReturnType<typeof mx.array>   // int32
  postIndices: ReturnType<typeof mx.array>  // int32
}

/**
 * Weighted connectivity - includes initial weights.
 */
export interface WeightedConnectivity extends Connectivity {
  weights: ReturnType<typeof mx.array>  // float32
}

// ============================================================================
// BASIC PATTERNS (consolidated from synapse module)
// ============================================================================

/**
 * All-to-all connectivity.
 * Every pre-neuron connects to every post-neuron.
 * Total synapses: preSize * postSize
 * ALL GPU operations.
 */
export function allToAll(
  preSize: number,
  postSize: number
): Connectivity {
  const totalSynapses = preSize * postSize

  // Pre indices: 0,0,0...1,1,1...2,2,2... (each pre connects to all post)
  const preIdx = mx.floor(mx.divide(mx.arange(totalSynapses), mx.array(postSize)))

  // Post indices: 0,1,2...0,1,2...0,1,2... (cycling through post neurons)
  const postIdx = mx.remainder(mx.arange(totalSynapses), mx.array(postSize))

  return {
    preIndices: preIdx.astype(mx.int32),
    postIndices: postIdx.astype(mx.int32),
  }
}

/**
 * One-to-one connectivity.
 * Pre-neuron i connects only to post-neuron i.
 * Requires preSize == postSize.
 * ALL GPU operations.
 */
export function oneToOne(size: number): Connectivity {
  const indices = mx.arange(size, mx.int32)
  return { preIndices: indices, postIndices: mx.array(indices) }
}

/**
 * Random sparse connectivity.
 * Each possible connection exists with probability = density.
 * ALL GPU operations.
 */
export function randomSparse(
  preSize: number,
  postSize: number,
  density: number
): Connectivity {
  const totalPossible = preSize * postSize

  // Generate random mask on GPU
  const mask = mx.less(mx.random.uniform(0, 1, [totalPossible]), mx.array(density))

  // Create index grids on GPU
  const preGrid = mx.floor(mx.divide(mx.arange(totalPossible), mx.array(postSize)))
  const postGrid = mx.remainder(mx.arange(totalPossible), mx.array(postSize))

  // Select only connected indices
  const preIdx = preGrid.index(mask)
  const postIdx = postGrid.index(mask)

  return {
    preIndices: preIdx.astype(mx.int32),
    postIndices: postIdx.astype(mx.int32),
  }
}

// ============================================================================
// BIOLOGICALLY IMPORTANT PATTERNS
// ============================================================================

/**
 * Lateral inhibition connectivity.
 * Each neuron inhibits its neighbors within a radius.
 * Used for contrast enhancement, winner-take-all.
 *
 * @param size - Population size (1D arrangement)
 * @param radius - How many neighbors to inhibit on each side
 * @param selfConnect - Whether to include self-connections (default false)
 * ALL GPU operations.
 */
export function lateralInhibition(
  size: number,
  radius: number,
  selfConnect: boolean = false
): Connectivity {
  const preList: number[] = []
  const postList: number[] = []

  // Build connectivity on CPU (structure), then convert to GPU
  for (let i = 0; i < size; i++) {
    for (let j = Math.max(0, i - radius); j <= Math.min(size - 1, i + radius); j++) {
      if (!selfConnect && i === j) continue
      preList.push(i)
      postList.push(j)
    }
  }

  return {
    preIndices: mx.array(preList, mx.int32),
    postIndices: mx.array(postList, mx.int32),
  }
}

/**
 * Center-surround connectivity (2D grid).
 * Each neuron excites itself and inhibits surrounding neurons.
 * Classic retinal ganglion cell pattern.
 *
 * @param width - Grid width
 * @param height - Grid height
 * @param excRadius - Radius of excitatory center
 * @param inhRadius - Radius of inhibitory surround
 * ALL GPU operations (structure built on CPU for simplicity).
 */
export function centerSurround(
  width: number,
  height: number,
  excRadius: number,
  inhRadius: number
): WeightedConnectivity {
  const preList: number[] = []
  const postList: number[] = []
  const weightList: number[] = []

  const size = width * height

  // Helper to convert 2D to 1D index
  const idx = (x: number, y: number) => y * width + x

  // Helper to compute distance
  const dist = (x1: number, y1: number, x2: number, y2: number) =>
    Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

  for (let y1 = 0; y1 < height; y1++) {
    for (let x1 = 0; x1 < width; x1++) {
      const pre = idx(x1, y1)

      for (let y2 = 0; y2 < height; y2++) {
        for (let x2 = 0; x2 < width; x2++) {
          const d = dist(x1, y1, x2, y2)
          const post = idx(x2, y2)

          if (d <= excRadius) {
            // Excitatory center
            preList.push(pre)
            postList.push(post)
            weightList.push(0.3)  // Positive weight
          } else if (d <= inhRadius) {
            // Inhibitory surround
            preList.push(pre)
            postList.push(post)
            weightList.push(-0.1)  // Negative weight (requires inhibitory neuron)
          }
        }
      }
    }
  }

  return {
    preIndices: mx.array(preList, mx.int32),
    postIndices: mx.array(postList, mx.int32),
    weights: mx.array(weightList, mx.float32),
  }
}

/**
 * Topographic mapping connectivity.
 * Spatial arrangement is preserved - nearby pre-neurons connect to nearby post-neurons.
 * Connection probability decreases with distance (Gaussian).
 *
 * @param preWidth - Pre-population grid width
 * @param preHeight - Pre-population grid height
 * @param postWidth - Post-population grid width
 * @param postHeight - Post-population grid height
 * @param sigma - Gaussian width (larger = more spread)
 * @param density - Base connection density
 * ALL GPU operations (structure built on CPU).
 */
export function topographic(
  preWidth: number,
  preHeight: number,
  postWidth: number,
  postHeight: number,
  sigma: number,
  density: number = 1.0
): Connectivity {
  const preList: number[] = []
  const postList: number[] = []

  const preSize = preWidth * preHeight
  const postSize = postWidth * postHeight

  // Scale factors for mapping coordinates
  const scaleX = postWidth / preWidth
  const scaleY = postHeight / preHeight

  for (let preY = 0; preY < preHeight; preY++) {
    for (let preX = 0; preX < preWidth; preX++) {
      const preIdx = preY * preWidth + preX

      // Map to post coordinates
      const centerX = preX * scaleX
      const centerY = preY * scaleY

      for (let postY = 0; postY < postHeight; postY++) {
        for (let postX = 0; postX < postWidth; postX++) {
          const postIdx = postY * postWidth + postX

          // Distance in post coordinates
          const dx = postX - centerX
          const dy = postY - centerY
          const dist2 = dx * dx + dy * dy

          // Gaussian probability
          const prob = Math.exp(-dist2 / (2 * sigma * sigma)) * density

          if (Math.random() < prob) {
            preList.push(preIdx)
            postList.push(postIdx)
          }
        }
      }
    }
  }

  return {
    preIndices: mx.array(preList, mx.int32),
    postIndices: mx.array(postList, mx.int32),
  }
}

/**
 * Recurrent connectivity within a population.
 * Neurons connect to each other with given density.
 * Used for working memory, attractor networks.
 *
 * @param size - Population size
 * @param density - Connection probability
 * @param selfConnect - Whether to include self-connections
 * ALL GPU operations.
 */
export function recurrent(
  size: number,
  density: number,
  selfConnect: boolean = false
): Connectivity {
  const conn = randomSparse(size, size, density)

  if (!selfConnect) {
    // Remove self-connections by filtering where pre != post
    // This requires CPU evaluation temporarily
    mx.eval(conn.preIndices, conn.postIndices)
    const preArr = conn.preIndices.tolist() as number[]
    const postArr = conn.postIndices.tolist() as number[]

    const filteredPre: number[] = []
    const filteredPost: number[] = []

    for (let i = 0; i < preArr.length; i++) {
      if (preArr[i] !== postArr[i]) {
        filteredPre.push(preArr[i])
        filteredPost.push(postArr[i])
      }
    }

    return {
      preIndices: mx.array(filteredPre, mx.int32),
      postIndices: mx.array(filteredPost, mx.int32),
    }
  }

  return conn
}

/**
 * Feedforward connectivity between layers.
 * Creates connectivity for a multi-layer feedforward network.
 *
 * @param layerSizes - Array of layer sizes [input, hidden1, hidden2, ..., output]
 * @param density - Connection density between layers
 * Returns array of Connectivity objects, one for each layer pair.
 */
export function feedforward(
  layerSizes: number[],
  density: number = 0.1
): Connectivity[] {
  const connections: Connectivity[] = []

  for (let i = 0; i < layerSizes.length - 1; i++) {
    connections.push(randomSparse(layerSizes[i], layerSizes[i + 1], density))
  }

  return connections
}

/**
 * Feedback connectivity (opposite of feedforward).
 * Higher layers project back to lower layers.
 *
 * @param layerSizes - Array of layer sizes [input, hidden1, ..., output]
 * @param density - Connection density
 * Returns array of Connectivity objects (output→hidden, hidden→input, etc.)
 */
export function feedback(
  layerSizes: number[],
  density: number = 0.05
): Connectivity[] {
  const connections: Connectivity[] = []

  for (let i = layerSizes.length - 1; i > 0; i--) {
    connections.push(randomSparse(layerSizes[i], layerSizes[i - 1], density))
  }

  return connections
}

// ============================================================================
// CIRCUIT TEMPLATES
// ============================================================================

/**
 * Circuit definition - multiple populations and their connections.
 */
export interface CircuitTemplate {
  populations: {
    id: string
    size: number
    excitatory: boolean
    type?: string
  }[]
  connections: {
    preId: string
    postId: string
    connectivity: Connectivity
    plastic?: boolean
  }[]
}

/**
 * Create a simple reflex arc circuit.
 * Sensory → Interneuron → Motor
 *
 * @param sensorSize - Number of sensory neurons
 * @param interSize - Number of interneurons
 * @param motorSize - Number of motor neurons
 * @param density - Connection density
 */
export function reflexArc(
  sensorSize: number,
  interSize: number,
  motorSize: number,
  density: number = 0.3
): CircuitTemplate {
  return {
    populations: [
      { id: 'sensory', size: sensorSize, excitatory: true, type: 'RS' },
      { id: 'inter', size: interSize, excitatory: true, type: 'RS' },
      { id: 'motor', size: motorSize, excitatory: true, type: 'RS' },
    ],
    connections: [
      {
        preId: 'sensory',
        postId: 'inter',
        connectivity: randomSparse(sensorSize, interSize, density),
        plastic: false,  // Innate wiring
      },
      {
        preId: 'inter',
        postId: 'motor',
        connectivity: randomSparse(interSize, motorSize, density),
        plastic: false,  // Innate wiring
      },
    ],
  }
}

/**
 * Create an oscillator circuit (CPG - Central Pattern Generator).
 * Two mutually inhibiting populations.
 *
 * @param size - Size of each population
 */
export function oscillator(size: number): CircuitTemplate {
  return {
    populations: [
      { id: 'exc1', size, excitatory: true, type: 'RS' },
      { id: 'inh1', size, excitatory: false, type: 'FS' },
      { id: 'exc2', size, excitatory: true, type: 'RS' },
      { id: 'inh2', size, excitatory: false, type: 'FS' },
    ],
    connections: [
      // Excitatory drives inhibitory (same side)
      { preId: 'exc1', postId: 'inh1', connectivity: allToAll(size, size) },
      { preId: 'exc2', postId: 'inh2', connectivity: allToAll(size, size) },
      // Inhibitory inhibits opposite excitatory
      { preId: 'inh1', postId: 'exc2', connectivity: allToAll(size, size) },
      { preId: 'inh2', postId: 'exc1', connectivity: allToAll(size, size) },
      // Recurrent excitation (maintains activity)
      { preId: 'exc1', postId: 'exc1', connectivity: recurrent(size, 0.3, false) },
      { preId: 'exc2', postId: 'exc2', connectivity: recurrent(size, 0.3, false) },
    ],
  }
}

/**
 * Create a winner-take-all circuit.
 * Excitatory neurons compete via shared inhibition.
 *
 * @param excSize - Number of excitatory (competing) neurons
 * @param inhSize - Number of inhibitory neurons
 */
export function winnerTakeAll(excSize: number, inhSize: number): CircuitTemplate {
  return {
    populations: [
      { id: 'exc', size: excSize, excitatory: true, type: 'RS' },
      { id: 'inh', size: inhSize, excitatory: false, type: 'FS' },
    ],
    connections: [
      // All excitatory drive all inhibitory
      { preId: 'exc', postId: 'inh', connectivity: allToAll(excSize, inhSize) },
      // All inhibitory inhibit all excitatory
      { preId: 'inh', postId: 'exc', connectivity: allToAll(inhSize, excSize) },
    ],
  }
}

/**
 * Create a simplified cortical column.
 * L4 (input) → L2/3 (processing) → L5 (output)
 * With inhibitory interneurons at each layer.
 *
 * @param l4Size - Layer 4 size (input from thalamus)
 * @param l23Size - Layer 2/3 size (processing)
 * @param l5Size - Layer 5 size (output)
 * @param inhRatio - Ratio of inhibitory neurons (default 0.2 = 20%)
 */
export function corticalColumn(
  l4Size: number,
  l23Size: number,
  l5Size: number,
  inhRatio: number = 0.2
): CircuitTemplate {
  const l4Inh = Math.floor(l4Size * inhRatio)
  const l23Inh = Math.floor(l23Size * inhRatio)
  const l5Inh = Math.floor(l5Size * inhRatio)

  return {
    populations: [
      // Layer 4
      { id: 'L4_exc', size: l4Size, excitatory: true, type: 'RS' },
      { id: 'L4_inh', size: l4Inh, excitatory: false, type: 'FS' },
      // Layer 2/3
      { id: 'L23_exc', size: l23Size, excitatory: true, type: 'RS' },
      { id: 'L23_inh', size: l23Inh, excitatory: false, type: 'FS' },
      // Layer 5
      { id: 'L5_exc', size: l5Size, excitatory: true, type: 'IB' },  // Intrinsically bursting
      { id: 'L5_inh', size: l5Inh, excitatory: false, type: 'FS' },
    ],
    connections: [
      // L4 → L2/3 (feedforward)
      { preId: 'L4_exc', postId: 'L23_exc', connectivity: randomSparse(l4Size, l23Size, 0.1) },
      { preId: 'L4_exc', postId: 'L23_inh', connectivity: randomSparse(l4Size, l23Inh, 0.1) },

      // L2/3 → L5 (feedforward)
      { preId: 'L23_exc', postId: 'L5_exc', connectivity: randomSparse(l23Size, l5Size, 0.05) },
      { preId: 'L23_exc', postId: 'L5_inh', connectivity: randomSparse(l23Size, l5Inh, 0.05) },

      // Local inhibition in each layer
      { preId: 'L4_inh', postId: 'L4_exc', connectivity: allToAll(l4Inh, l4Size) },
      { preId: 'L23_inh', postId: 'L23_exc', connectivity: allToAll(l23Inh, l23Size) },
      { preId: 'L5_inh', postId: 'L5_exc', connectivity: allToAll(l5Inh, l5Size) },

      // Excitatory → local inhibitory
      { preId: 'L4_exc', postId: 'L4_inh', connectivity: randomSparse(l4Size, l4Inh, 0.3) },
      { preId: 'L23_exc', postId: 'L23_inh', connectivity: randomSparse(l23Size, l23Inh, 0.3) },
      { preId: 'L5_exc', postId: 'L5_inh', connectivity: randomSparse(l5Size, l5Inh, 0.3) },

      // Recurrent in L2/3 (working memory)
      { preId: 'L23_exc', postId: 'L23_exc', connectivity: recurrent(l23Size, 0.1, false) },
    ],
  }
}
