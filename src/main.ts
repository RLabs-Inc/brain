#!/usr/bin/env bun
import { mount } from 'sveltui'
import { mount as mountComponent } from 'svelte'
import App from './App.svelte'

mount(() => {
  mountComponent(App, {
    target: document.body
  })
}, { fullscreen: true })
