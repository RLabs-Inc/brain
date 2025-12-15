import { core as mx } from '@frost-beta/mlx'

console.log('=== Testing Building Blocks ===\n')

// Import our building blocks
import {
  allocatePopulation,
  voltage,
  current,
  populationSize,
  integrate,
  THRESHOLD,
} from './dist/src/core/neuron.svelte.js'

import {
  allocateSynapseGroup,
  createAllToAllConnectivity,
  weights,
  transmit,
} from './dist/src/core/synapse.svelte.js'

import {
  allocateNetwork,
  addPopulationToNetwork,
  addSynapseGroupToNetwork,
  step,
} from './dist/src/core/network.svelte.js'

async function runTest() {
  try {
    console.log('1. Creating populations...')
    const pop1 = allocatePopulation('input', 5, 'RS')
    const pop2 = allocatePopulation('output', 5, 'RS')
    console.log('   Created pop1=' + pop1 + ', pop2=' + pop2)
    console.log('   Sizes: ' + populationSize[pop1] + ', ' + populationSize[pop2])

    console.log('\n2. Checking initial voltage...')
    await mx.asyncEval(voltage[pop1])
    const avgV1 = mx.mean(voltage[pop1])
    await mx.asyncEval(avgV1)
    console.log('   Pop1 avg voltage: ' + avgV1.item())

    console.log('\n3. Creating network...')
    const net = allocateNetwork('test')
    addPopulationToNetwork(net, pop1)
    addPopulationToNetwork(net, pop2)
    console.log('   Network ' + net + ' created')

    console.log('\n4. Creating synapses...')
    const conn = createAllToAllConnectivity(5, 5)
    await mx.asyncEval(conn.preIndices, conn.postIndices)
    console.log('   Connectivity shape: pre=' + conn.preIndices.shape + ', post=' + conn.postIndices.shape)

    const syn = allocateSynapseGroup('syn1', pop1, pop2, conn.preIndices, conn.postIndices)
    addSynapseGroupToNetwork(net, syn)
    await mx.asyncEval(weights[syn])
    const avgW = mx.mean(weights[syn])
    await mx.asyncEval(avgW)
    console.log('   Synapse group ' + syn + ', avg weight: ' + avgW.item())

    console.log('\n5. Injecting current...')
    const inputCurrent = mx.multiply(mx.random.uniform(0, 1, [5]), mx.array(20))
    current[pop1] = mx.add(current[pop1], inputCurrent)
    await mx.asyncEval(current[pop1])
    const avgCurrent = mx.mean(current[pop1])
    await mx.asyncEval(avgCurrent)
    console.log('   Injected current, avg: ' + avgCurrent.item())

    console.log('\n6. Running simulation step...')
    step(net, 1.0)
    await mx.asyncEval(voltage[pop1], voltage[pop2])
    const newAvgV1 = mx.mean(voltage[pop1])
    const newAvgV2 = mx.mean(voltage[pop2])
    await mx.asyncEval(newAvgV1, newAvgV2)
    console.log('   After step - Pop1 avg V: ' + newAvgV1.item() + ', Pop2 avg V: ' + newAvgV2.item())

    console.log('\n7. Checking for spikes...')
    const firing1 = mx.greaterEqual(voltage[pop1], mx.array(THRESHOLD))
    const firing2 = mx.greaterEqual(voltage[pop2], mx.array(THRESHOLD))
    const spikes1 = mx.sum(firing1)
    const spikes2 = mx.sum(firing2)
    await mx.asyncEval(spikes1, spikes2)
    console.log('   Pop1 spikes: ' + spikes1.item() + ', Pop2 spikes: ' + spikes2.item())

    console.log('\n8. Running more steps to see activity...')
    for (let i = 0; i < 10; i++) {
      // Inject more current
      const noise = mx.multiply(mx.random.uniform(0, 1, [5]), mx.array(15))
      current[pop1] = mx.add(current[pop1], noise)

      // Step
      step(net, 1.0)

      // Check spikes
      await mx.asyncEval(voltage[pop1], voltage[pop2])
      const f1 = mx.greaterEqual(voltage[pop1], mx.array(THRESHOLD))
      const f2 = mx.greaterEqual(voltage[pop2], mx.array(THRESHOLD))
      const s1 = mx.sum(f1)
      const s2 = mx.sum(f2)
      await mx.asyncEval(s1, s2)
      console.log('   Step ' + (i+2) + ': Pop1=' + s1.item() + ' spikes, Pop2=' + s2.item() + ' spikes')
    }

    console.log('\n=== Test Complete ===')
  } catch (e) {
    console.error('ERROR:', e)
  }
}

runTest()
