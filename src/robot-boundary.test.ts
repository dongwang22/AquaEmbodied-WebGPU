import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fluidBoundaryConditions} from './boundary-topology';

test('robot pool declares periodic lateral faces and solid vertical faces',()=>{
  assert.deepEqual(fluidBoundaryConditions(),[
    'periodic','periodic','periodic','periodic','solid-wall','solid-wall',
  ]);
});

test('production WGSL wraps robot x/y neighbors and only seals robot z faces',()=>{
  const source=readFileSync(new URL('./shaders/common.wgsl',import.meta.url),'utf8');
  assert.match(source,/periodicCoordinate\(q\.x,dims\.x\)/);
  assert.match(source,/periodicCoordinate\(q\.y,dims\.y\)/);
  assert.match(source,/clamp\(q\.z,0,dims\.z-1\)/);
  assert.match(source,/return q\.z<=0\|\|q\.z>=i32\(p\.grid\.z\)-1;/);
  assert.doesNotMatch(source,/ROBOT_POOL|river|bed=/i);
});
