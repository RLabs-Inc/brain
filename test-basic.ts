import { core as mx } from '@frost-beta/mlx'

console.log('Testing MLX basics...')

// Test 1: Create array
const arr = mx.array([1, 2, 3])
console.log('Array created:', arr.shape)

// Test 2: Evaluate
mx.eval(arr)
console.log('Array evaluated')

// Test 3: Sum (returns scalar)
const sum = mx.sum(arr)
mx.eval(sum)
console.log('Sum:', sum.item())

// Test 4: Create scalar
const scalar = mx.array(42)
mx.eval(scalar)
console.log('Scalar value:', scalar.item())

// Test 5: GPU operations
const a = mx.array([1, 2, 3])
const b = mx.array([4, 5, 6])
const c = mx.add(a, b)
mx.eval(c)
console.log('Add result shape:', c.shape)
const cSum = mx.sum(c)
mx.eval(cSum)
console.log('Add sum:', cSum.item())

console.log('MLX basics work!')
