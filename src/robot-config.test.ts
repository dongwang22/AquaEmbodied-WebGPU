import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {readRobotControlPresets,readRobotParameters} from './robot-config';
import {robotFluidConfig} from './robot-config';
import {ROBOT_MAX_WATER_LEVEL} from './robot-motion';

const raw=()=>JSON.parse(readFileSync(new URL('../public/robotInputParameter.json',import.meta.url),'utf8'));

test('robot JSON owns the three UI preset values and selected defaults',()=>{
  const value=raw(),presets=readRobotControlPresets(value),parameters=readRobotParameters(value);
  assert.deepEqual(presets.viscosity,value.viscosity);
  assert.deepEqual(presets.surfaceTension,value.surfaceTension);
  assert.equal(parameters.viscosity,value.viscosity[value.viscosity.default]);
  assert.equal(parameters.surfaceTension,value.surfaceTension[value.surfaceTension.default]);
  for(const group of [presets.viscosity,presets.surfaceTension]){
    assert.equal(group.small,group.medium*.2);
    assert.equal(group.large,group.medium*5);
    assert.equal(group.default,'medium');
  }
});

test('legacy numeric robot fluid values and explicit UI overrides remain supported',()=>{
  const value=raw();
  assert.equal(readRobotParameters({...value,viscosity:.02,surfaceTension:.003}).viscosity,.02);
  assert.equal(readRobotParameters({...value,viscosity:.02,surfaceTension:.003}).surfaceTension,.003);
  assert.throws(()=>readRobotParameters({...value,viscosity:{...value.viscosity,small:0}}));
  assert.throws(()=>readRobotParameters({...value,surfaceTension:{...value.surfaceTension,default:'custom'}}));
});

test('balanced robot refinement maps the centered pool to 47 by 47 by 54',()=>{
  const parameters=readRobotParameters({...raw(),gridScale:.4375});
  const config=robotFluidConfig(parameters);
  assert.deepEqual([config.nx,config.ny,config.nz],[47,47,54]);
  assert.throws(()=>readRobotParameters({...raw(),gridScale:.4}));
});

test('high is the centered 60 by 60 by 70 robot case',()=>{
  const parameters=readRobotParameters({...raw(),gridScale:.5625});
  const config=robotFluidConfig(parameters);
  assert.deepEqual([config.nx,config.ny,config.nz],[60,60,70]);
  assert.throws(()=>readRobotParameters({...raw(),gridScale:.75}));
});

test('water level accepts exactly seventy percent of the domain height',()=>{
  const parameters=readRobotParameters({...raw(),waterLevel:ROBOT_MAX_WATER_LEVEL});
  const config=robotFluidConfig(parameters);
  assert.equal(config.waterLevel!/ROBOT_DOMAIN_HEIGHT,.7);
  assert.throws(()=>readRobotParameters({...raw(),waterLevel:ROBOT_MAX_WATER_LEVEL+.001}));
});

const ROBOT_DOMAIN_HEIGHT=ROBOT_MAX_WATER_LEVEL/.7;

