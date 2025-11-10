/**
 * three.js - OrbitControls (UMD, r158-ish)
 * https://threejs.org
 * 
 * MIT License
 * (c) authors of three.js / mrdoob
 */
(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD
    define(['three'], factory);
  } else if (typeof exports === 'object') {
    // Node / CommonJS
    module.exports = factory(require('three'));
  } else {
    // Browser globals (root is window)
    root.THREE = root.THREE || {};
    root.THREE.OrbitControls = factory(root.THREE);
  }
}(typeof self !== 'undefined' ? self : this, function (THREE) {
  if (!THREE) throw new Error('OrbitControls: THREE not found in global scope');

  var _changeEvent = { type: 'change' };
  var _startEvent  = { type: 'start'  };
  var _endEvent    = { type: 'end'    };

  var STATE = {
    NONE: -1, ROTATE: 0, DOLLY: 1, PAN: 2,
    TOUCH_ROTATE: 3, TOUCH_PAN: 4, TOUCH_DOLLY_PAN: 5, TOUCH_DOLLY_ROTATE: 6
  };

  function OrbitControls(object, domElement) {
    if (!object) throw new Error('OrbitControls: camera is required');

    THREE.EventDispatcher.call(this);

    this.object = object;
    this.domElement = (domElement !== undefined) ? domElement : document;

    // configurable
    this.enabled = true;

    // focus target
    this.target = new THREE.Vector3();

    // distance limits (persp)
    this.minDistance = 0;
    this.maxDistance = Infinity;

    // zoom limits (ortho)
    this.minZoom = 0;
    this.maxZoom = Infinity;

    // spherical limits
    this.minPolarAngle = 0;
    this.maxPolarAngle = Math.PI;
    this.minAzimuthAngle = -Infinity;
    this.maxAzimuthAngle = Infinity;

    // inertia
    this.enableDamping = true;
    this.dampingFactor = 0.05;

    // zoom
    this.enableZoom = true;
    this.zoomSpeed = 1.0;

    // rotate
    this.enableRotate = true;
    this.rotateSpeed = 1.0;

    // pan
    this.enablePan = true;
    this.panSpeed = 1.0;
    this.screenSpacePanning = true;
    this.keyPanSpeed = 7.0;

    // auto-rotate
    this.autoRotate = false;
    this.autoRotateSpeed = 2.0;

    // inputs
    this.keys = { LEFT: 'ArrowLeft', UP: 'ArrowUp', RIGHT: 'ArrowRight', BOTTOM: 'ArrowDown' };
    this.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };

    // state
    var scope = this;

    var state = STATE.NONE;

    var EPS = 1e-6;

    var spherical = new THREE.Spherical();
    var sphericalDelta = new THREE.Spherical();

    var scale = 1;
    var panOffset = new THREE.Vector3();

    var rotateStart = new THREE.Vector2();
    var rotateEnd   = new THREE.Vector2();
    var rotateDelta = new THREE.Vector2();

    var panStart = new THREE.Vector2();
    var panEnd   = new THREE.Vector2();
    var panDelta = new THREE.Vector2();

    var dollyStart = new THREE.Vector2();
    var dollyEnd   = new THREE.Vector2();
    var dollyDelta = new THREE.Vector2();

    // for reset
    this.target0   = this.target.clone();
    this.position0 = this.object.position.clone();
    this.zoom0     = this.object.zoom;

    // methods
    this.getPolarAngle = function () { return spherical.phi; };
    this.getAzimuthalAngle = function () { return spherical.theta; };

    this.saveState = function () {
      scope.target0.copy(scope.target);
      scope.position0.copy(scope.object.position);
      scope.zoom0 = scope.object.zoom;
    };

    this.reset = function () {
      scope.target.copy(scope.target0);
      scope.object.position.copy(scope.position0);
      scope.object.zoom = scope.zoom0;
      scope.object.updateProjectionMatrix();
      scope.dispatchEvent(_changeEvent);
      scope.update();
      state = STATE.NONE;
    };

    this.update = (function () {
      var offset = new THREE.Vector3();
      var lastPosition = new THREE.Vector3();
      var lastQuaternion = new THREE.Quaternion();
      var lastTarget = new THREE.Vector3();

      var quat = new THREE.Quaternion().setFromUnitVectors(scope.object.up, new THREE.Vector3(0, 1, 0));
      var quatInverse = quat.clone().invert();

      return function update() {
        var position = scope.object.position;

        offset.copy(position).sub(scope.target);
        offset.applyQuaternion(quat);
        spherical.setFromVector3(offset);

        if (scope.autoRotate && state === STATE.NONE) {
          rotateLeft(2 * Math.PI / 60 / 60 * scope.autoRotateSpeed);
        }

        if (scope.enableDamping) {
          spherical.theta += sphericalDelta.theta * scope.dampingFactor;
          spherical.phi += sphericalDelta.phi * scope.dampingFactor;
        } else {
          spherical.theta += sphericalDelta.theta;
          spherical.phi += sphericalDelta.phi;
        }

        // clamp azimuth
        var min = scope.minAzimuthAngle, max = scope.maxAzimuthAngle, twoPI = 2 * Math.PI;
        if (isFinite(min) && isFinite(max)) {
          if (min < -Math.PI) min += twoPI; else if (min > Math.PI) min -= twoPI;
          if (max < -Math.PI) max += twoPI; else if (max > Math.PI) max -= twoPI;
          if (min <= max) spherical.theta = Math.max(min, Math.min(max, spherical.theta));
          else spherical.theta = (spherical.theta > (min + max) / 2) ? Math.max(min, spherical.theta) : Math.min(max, spherical.theta);
        }

        // clamp polar
        spherical.phi = Math.max(scope.minPolarAngle, Math.min(scope.maxPolarAngle, spherical.phi));
        spherical.makeSafe();

        // pan
        if (scope.enableDamping) panOffset.multiplyScalar(1 - scope.dampingFactor);
        scope.target.add(panOffset);

        // dolly
        spherical.radius *= scale;
        spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));
        scale = 1;

        offset.setFromSpherical(spherical);
        offset.applyQuaternion(quatInverse);

        position.copy(scope.target).add(offset);
        scope.object.lookAt(scope.target);

        if (scope.enableDamping) {
          sphericalDelta.theta *= (1 - scope.dampingFactor);
          sphericalDelta.phi *= (1 - scope.dampingFactor);
        } else {
          sphericalDelta.set(0, 0, 0);
          panOffset.set(0, 0, 0);
        }

        if (
          lastPosition.distanceToSquared(scope.object.position) > EPS ||
          8 * (1 - lastQuaternion.dot(scope.object.quaternion)) > EPS ||
          lastTarget.distanceToSquared(scope.target) > 0
        ) {
          scope.dispatchEvent(_changeEvent);
          lastPosition.copy(scope.object.position);
          lastQuaternion.copy(scope.object.quaternion);
          lastTarget.copy(scope.target);
          return true;
        }
        return false;
      };
    }());

    // internals
    var onContextMenu = function (event) { if (scope.enabled === false) return; event.preventDefault(); };

    var onMouseDown = function (event) {
      if (scope.enabled === false) return;
      scope.domElement.setPointerCapture && scope.domElement.setPointerCapture(event.pointerId);

      switch (event.button) {
        case 0: // LEFT
          if (event.ctrlKey || event.metaKey || event.shiftKey) {
            if (scope.enablePan === false) return;
            handleMouseDownPan(event); state = STATE.PAN;
          } else {
            if (scope.enableRotate === false) return;
            handleMouseDownRotate(event); state = STATE.ROTATE;
          }
          break;
        case 1: // MIDDLE
          if (scope.enableZoom === false) return;
          handleMouseDownDolly(event); state = STATE.DOLLY; break;
        case 2: // RIGHT
          if (scope.enablePan === false) return;
          handleMouseDownPan(event); state = STATE.PAN; break;
      }
      if (state !== STATE.NONE) scope.dispatchEvent(_startEvent);
    };

    var onMouseMove = function (event) {
      if (scope.enabled === false) return;
      switch (state) {
        case STATE.ROTATE: if (scope.enableRotate === false) return; handleMouseMoveRotate(event); break;
        case STATE.DOLLY:  if (scope.enableZoom   === false) return; handleMouseMoveDolly(event);  break;
        case STATE.PAN:    if (scope.enablePan    === false) return; handleMouseMovePan(event);    break;
      }
    };

    var onMouseUp = function () {
      if (scope.enabled === false) return;
      scope.dispatchEvent(_endEvent);
      state = STATE.NONE;
    };

    var onMouseWheel = function (event) {
      if (scope.enabled === false || scope.enableZoom === false || state !== STATE.NONE) return;
      event.preventDefault();
      scope.dispatchEvent(_startEvent);
      handleMouseWheel(event);
      scope.dispatchEvent(_endEvent);
    };

    var onKeyDown = function (event) {
      if (scope.enabled === false || scope.enablePan === false) return;
      var needsUpdate = false;
      switch (event.code) {
        case scope.keys.UP:     pan(0,  scope.keyPanSpeed); needsUpdate = true; break;
        case scope.keys.BOTTOM: pan(0, -scope.keyPanSpeed); needsUpdate = true; break;
        case scope.keys.LEFT:   pan( scope.keyPanSpeed, 0); needsUpdate = true; break;
        case scope.keys.RIGHT:  pan(-scope.keyPanSpeed, 0); needsUpdate = true; break;
      }
      if (needsUpdate) { event.preventDefault(); scope.update(); }
    };

    // handlers
    function handleMouseDownRotate(event) { rotateStart.set(event.clientX, event.clientY); }
    function handleMouseDownDolly (event) { dollyStart .set(event.clientX, event.clientY); }
    function handleMouseDownPan   (event) { panStart   .set(event.clientX, event.clientY); }

    function handleMouseMoveRotate(event) {
      rotateEnd.set(event.clientX, event.clientY);
      rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
      var element = scope.domElement;
      rotateLeft( 2 * Math.PI * rotateDelta.x / element.clientHeight );
      rotateUp  ( 2 * Math.PI * rotateDelta.y / element.clientHeight );
      rotateStart.copy(rotateEnd);
      scope.update();
    }

    function handleMouseMoveDolly(event) {
      dollyEnd.set(event.clientX, event.clientY);
      dollyDelta.subVectors(dollyEnd, dollyStart);
      if (dollyDelta.y > 0) dollyOut(getZoomScale());
      else if (dollyDelta.y < 0) dollyIn(getZoomScale());
      dollyStart.copy(dollyEnd);
      scope.update();
    }

    function handleMouseMovePan(event) {
      panEnd.set(event.clientX, event.clientY);
      panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
      pan(panDelta.x, panDelta.y);
      panStart.copy(panEnd);
      scope.update();
    }

    function handleMouseWheel(event) {
      if (event.deltaY < 0) dollyIn(getZoomScale());
      else if (event.deltaY > 0) dollyOut(getZoomScale());
      scope.update();
    }

    function getZoomScale() { return Math.pow(0.95, scope.zoomSpeed); }

    function rotateLeft(angle) { sphericalDelta.theta -= angle; }
    function rotateUp  (angle) { sphericalDelta.phi   -= angle; }

    function panLeft(distance, objectMatrix) {
      var v = new THREE.Vector3();
      v.setFromMatrixColumn(objectMatrix, 0);
      v.multiplyScalar(-distance);
      panOffset.add(v);
    }

    function panUp(distance, objectMatrix) {
      var v = new THREE.Vector3();
      if (scope.screenSpacePanning === true) v.setFromMatrixColumn(objectMatrix, 1);
      else { v.setFromMatrixColumn(objectMatrix, 0); v.crossVectors(scope.object.up, v); }
      v.multiplyScalar(distance);
      panOffset.add(v);
    }

    function pan(deltaX, deltaY) {
      var element = scope.domElement;
      if (scope.object.isPerspectiveCamera) {
        var position = scope.object.position;
        var offset = new THREE.Vector3().copy(position).sub(scope.target);
        var targetDistance = offset.length();
        targetDistance *= Math.tan((scope.object.fov / 2) * Math.PI / 180.0);
        panLeft( 2 * deltaX * targetDistance / element.clientHeight, scope.object.matrix );
        panUp  ( 2 * deltaY * targetDistance / element.clientHeight, scope.object.matrix );
      } else if (scope.object.isOrthographicCamera) {
        panLeft( deltaX * (scope.object.right - scope.object.left) / scope.object.zoom / element.clientWidth, scope.object.matrix );
        panUp  ( deltaY * (scope.object.top - scope.object.bottom) / scope.object.zoom / element.clientHeight, scope.object.matrix );
      } else {
        scope.enablePan = false;
      }
    }

    function dollyOut(dollyScale) {
      if (scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera) scale /= dollyScale;
      else scope.enableZoom = false;
    }

    function dollyIn(dollyScale) {
      if (scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera) scale *= dollyScale;
      else scope.enableZoom = false;
    }

    // listeners
    scope.domElement.addEventListener('contextmenu', onContextMenu);
    scope.domElement.addEventListener('mousedown', onMouseDown);
    scope.domElement.addEventListener('wheel', onMouseWheel, { passive: false });
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);

    // init
    this.update();
  }

  // inherit
  OrbitControls.prototype = Object.create(THREE.EventDispatcher.prototype);
  OrbitControls.prototype.constructor = OrbitControls;

  return OrbitControls;
}));
