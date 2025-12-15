<script lang="ts">
  import { Box, Text } from 'sveltui'
  import { keyboard } from 'sveltui'
  import { core as mx } from '@frost-beta/mlx'
  import { onMount, onDestroy } from 'svelte'

  // Import our building blocks
  import {
    allocatePopulation,
    voltage,
    current,
    populationSize,
    getNeuronDerived,
    THRESHOLD,
  } from './core/neuron.svelte.ts'

  import {
    allocateSynapseGroup,
    createAllToAllConnectivity,
    weights,
    getSynapseDerived,
  } from './core/synapse.svelte.ts'

  import {
    allocateNetwork,
    addPopulationToNetwork,
    addSynapseGroupToNetwork,
    step,
    getNetworkDerived,
    setRewardValue,
  } from './core/network.svelte.ts'

  // Test state
  let networkIndex = $state(-1)
  let pop1Index = $state(-1)
  let pop2Index = $state(-1)
  let synapseIndex = $state(-1)
  let initialized = $state(false)
  let stepCount = $state(0)

  // Auto-run state
  let autoRun = $state(false)
  let autoRunInterval: ReturnType<typeof setInterval> | null = null

  // Cumulative stats
  let totalInputSpikes = $state(0)
  let totalOutputSpikes = $state(0)
  let initialWeight = $state(0)

  // Stats to display (will be updated from GPU)
  let stats = $state({
    pop1Spikes: 0,
    pop2Spikes: 0,
    pop1AvgV: '-70.0',
    pop2AvgV: '-70.0',
    avgWeight: '0.500',
    totalSpikes: 0,
  })

  // Status messages
  let status = $state('Initializing...')
  let error = $state('')

  // Helper to generate activity bar
  function generateActivityBar(spikes: number, maxNeurons: number): string {
    const filled = Math.min(spikes, maxNeurons)
    const empty = maxNeurons - filled
    return '█'.repeat(filled) + '░'.repeat(empty)
  }

  // Initialize the test network
  async function initNetwork() {
    try {
      status = 'Creating populations...'

      // Create two small populations (10 neurons each for testing)
      pop1Index = allocatePopulation('input', 10, 'RS')
      pop2Index = allocatePopulation('output', 10, 'RS')

      status = 'Creating network...'

      // Create network and add populations
      networkIndex = allocateNetwork('test')
      addPopulationToNetwork(networkIndex, pop1Index)
      addPopulationToNetwork(networkIndex, pop2Index)

      status = 'Creating synapses...'

      // Create all-to-all connectivity (GPU arrays)
      const conn = createAllToAllConnectivity(10, 10)

      // Create synapse group with stronger initial weights
      synapseIndex = allocateSynapseGroup(
        'input_to_output',
        pop1Index,
        pop2Index,
        conn.preIndices,
        conn.postIndices,
        { minWeight: 0.5, maxWeight: 5.0 }
      )
      addSynapseGroupToNetwork(networkIndex, synapseIndex)

      // Force GPU evaluation to confirm initialization
      await mx.asyncEval(voltage[pop1Index], voltage[pop2Index], weights[synapseIndex])

      // Store initial weight for comparison
      const avgW = mx.mean(weights[synapseIndex])
      await mx.asyncEval(avgW)
      initialWeight = avgW.item() as number

      initialized = true
      status = 'Ready! [SPACE]=Step [R]=Auto [+/-]=Reward [Q]=Quit'

      // Get initial stats
      await updateStats()

    } catch (e: any) {
      error = `Init error: ${e?.message || e}`
      status = 'Failed'
    }
  }

  // Run one simulation step
  async function runStep() {
    if (!initialized) return

    try {
      // Inject VERY STRONG current to make neurons fire reliably
      // Izhikevich with dt=1ms needs ~100+ current to jump from -70mV to 30mV threshold
      // Base 80 + random 0-40 = 80-120, ensuring spikes
      const baseCurrent = mx.full([populationSize[pop1Index]], 80, mx.float32)
      const noise = mx.multiply(
        mx.random.uniform(0, 1, [populationSize[pop1Index]]),
        mx.array(40)
      )
      current[pop1Index] = mx.add(baseCurrent, noise)

      // Run network step
      step(networkIndex, 1.0)
      stepCount++

      // Force evaluation
      await mx.asyncEval(voltage[pop1Index], voltage[pop2Index])

      // Update stats
      await updateStats()

      // Accumulate spikes
      totalInputSpikes += stats.pop1Spikes
      totalOutputSpikes += stats.pop2Spikes

      if (!autoRun) {
        status = `Step ${stepCount} | [SPACE]=Step [R]=Auto [+/-]=Reward [Q]=Quit`
      }

    } catch (e: any) {
      error = `Step error: ${e?.message || e}`
      status = 'Step failed'
    }
  }

  // Toggle auto-run mode
  function toggleAutoRun() {
    autoRun = !autoRun
    if (autoRun) {
      status = `AUTO-RUN | Step ${stepCount} | [R]=Stop [+/-]=Reward [Q]=Quit`
      autoRunInterval = setInterval(runStep, 100) // 10 steps per second
    } else {
      if (autoRunInterval) clearInterval(autoRunInterval)
      autoRunInterval = null
      status = `Step ${stepCount} | [SPACE]=Step [R]=Auto [+/-]=Reward [Q]=Quit`
    }
  }

  // Inject reward signal
  function injectReward(positive: boolean) {
    if (!initialized) return
    const rewardValue = positive ? 1.0 : -0.5
    setRewardValue(networkIndex, rewardValue)
    status = `${positive ? '+' : '-'} Reward injected! Step ${stepCount}`
  }

  // Update display stats from GPU
  async function updateStats() {
    try {
      // Get derived values
      const pop1Derived = getNeuronDerived(pop1Index)()
      const pop2Derived = getNeuronDerived(pop2Index)()
      const synDerived = getSynapseDerived(synapseIndex)()
      const netDerived = getNetworkDerived(networkIndex)()

      // Force GPU evaluation
      await mx.asyncEval(
        pop1Derived.spikeCount,
        pop2Derived.spikeCount,
        pop1Derived.avgVoltage,
        pop2Derived.avgVoltage,
        synDerived.avgWeight,
        netDerived.totalSpikes
      )

      // Now read values (only place we touch CPU!)
      stats = {
        pop1Spikes: pop1Derived.spikeCount.item() as number,
        pop2Spikes: pop2Derived.spikeCount.item() as number,
        pop1AvgV: (pop1Derived.avgVoltage.item() as number).toFixed(1),
        pop2AvgV: (pop2Derived.avgVoltage.item() as number).toFixed(1),
        avgWeight: (synDerived.avgWeight.item() as number).toFixed(3),
        totalSpikes: netDerived.totalSpikes.item() as number,
      }
    } catch (e: any) {
      error = `Stats error: ${e?.message || e}`
    }
  }

  // Initialize on mount
  onMount(() => {
    initNetwork()

    // Set up keyboard controls
    keyboard.onKey(' ', () => { if (!autoRun) runStep() })  // Space key
    keyboard.onKey('Space', () => { if (!autoRun) runStep() })  // Also try this variant
    keyboard.onKey('r', toggleAutoRun)
    keyboard.onKey('R', toggleAutoRun)
    keyboard.onKey('=', () => injectReward(true))  // + key
    keyboard.onKey('+', () => injectReward(true))
    keyboard.onKey('-', () => injectReward(false))
    keyboard.onKey('q', () => process.exit(0))
    keyboard.onKey('Q', () => process.exit(0))
  })

  onDestroy(() => {
    if (autoRunInterval) clearInterval(autoRunInterval)
  })
</script>

<Box width="100%" height="100%" flexDirection="column" padding={1}>
  <!-- Header -->
  <Box width="100%" borderBottom={1} paddingBottom={1} marginBottom={1}>
    <Text text="🧠 Vessel Brain Simulator - Building Block Test" color={0x00ff88} bold />
  </Box>

  <!-- Status -->
  <Box width="100%" marginBottom={1}>
    <Text text={`Status: ${status}`} color={0xffff00} />
  </Box>

  {#if error}
    <Box width="100%" marginBottom={1}>
      <Text text={`Error: ${error}`} color={0xff0000} bold />
    </Box>
  {/if}

  <!-- Main content -->
  {#if initialized}
    <Box width="100%" flexDirection="row" gap={2}>
      <!-- Left panel: Network info -->
      <Box width="50%" flexDirection="column" border={1} padding={1}>
        <Text text="Network Stats" color={0x00ffff} bold />
        <Text text="─────────────────────" color={0x444444} />
        <Text text={`Step: ${stepCount}`} />
        <Text text={`This Step Spikes: ${stats.totalSpikes}`} color={stats.totalSpikes > 0 ? 0xff8800 : 0x888888} />
        <Text text={`Total Input Spikes: ${totalInputSpikes}`} color={0x00ff00} />
        <Text text={`Total Output Spikes: ${totalOutputSpikes}`} color={0xff8800} />
      </Box>

      <!-- Right panel: Population info -->
      <Box width="50%" flexDirection="column" border={1} padding={1}>
        <Text text="Populations" color={0x00ffff} bold />
        <Text text="─────────────────────" color={0x444444} />
        <Text text={`Input (${populationSize[pop1Index]} neurons):`} />
        <Text text={`  Now: ${stats.pop1Spikes} spikes  V: ${stats.pop1AvgV}mV`} color={stats.pop1Spikes > 0 ? 0x00ff00 : 0x888888} />
        <Text text={`Output (${populationSize[pop2Index]} neurons):`} />
        <Text text={`  Now: ${stats.pop2Spikes} spikes  V: ${stats.pop2AvgV}mV`} color={stats.pop2Spikes > 0 ? 0x00ff00 : 0x888888} />
      </Box>
    </Box>

    <!-- Activity visualization -->
    <Box width="100%" marginTop={1} flexDirection="column" border={1} padding={1}>
      <Text text="Live Activity (this step):" color={0x00ffff} bold />
      <Box flexDirection="row" gap={1}>
        <Text text="Input:  " />
        <Text text={generateActivityBar(stats.pop1Spikes, 10)} color={0x00ff00} />
        <Text text={` ${stats.pop1Spikes}/10`} color={0x888888} />
      </Box>
      <Box flexDirection="row" gap={1}>
        <Text text="Output: " />
        <Text text={generateActivityBar(stats.pop2Spikes, 10)} color={0xff8800} />
        <Text text={` ${stats.pop2Spikes}/10`} color={0x888888} />
      </Box>
    </Box>

    <!-- Synapse info -->
    <Box width="100%" marginTop={1} flexDirection="column" border={1} padding={1}>
      <Text text="Synapses (100 connections) - STDP Learning:" color={0x00ffff} bold />
      <Box flexDirection="row" gap={2}>
        <Text text={`Initial Weight: ${initialWeight.toFixed(3)}`} color={0x888888} />
        <Text text={`Current Weight: ${stats.avgWeight}`} color={Number(stats.avgWeight) !== initialWeight ? 0x00ff00 : 0xffffff} />
        <Text text={`Change: ${(Number(stats.avgWeight) - initialWeight).toFixed(4)}`} color={Number(stats.avgWeight) > initialWeight ? 0x00ff00 : (Number(stats.avgWeight) < initialWeight ? 0xff4444 : 0x888888)} />
      </Box>
      <Text text="  Press [+] for reward, [-] for punishment" color={0x666666} />
    </Box>

    <!-- Mode indicator -->
    {#if autoRun}
      <Box width="100%" marginTop={1} justifyContent="center">
        <Text text=">>> AUTO-RUN MODE <<< Press [R] to stop" color={0xff00ff} bold />
      </Box>
    {/if}
  {:else}
    <Box width="100%" height="50%" justifyContent="center" alignItems="center">
      <Text text="Loading..." color={0x888888} />
    </Box>
  {/if}
</Box>
