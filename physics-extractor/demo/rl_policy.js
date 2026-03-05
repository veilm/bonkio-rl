(function (global) {
  const ACTION_TO_INTENT = {
    0: { left: false, right: false, up: false, down: false },
    1: { left: true, right: false, up: false, down: false },
    2: { left: false, right: true, up: false, down: false },
    3: { left: false, right: false, up: true, down: false },
    4: { left: false, right: false, up: false, down: true },
    5: { left: true, right: false, up: true, down: false },
    6: { left: false, right: true, up: true, down: false },
    7: { left: true, right: false, up: false, down: true },
    8: { left: false, right: true, up: false, down: true },
  };

  function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e1 = Math.exp(x);
    const e2 = Math.exp(-x);
    return (e1 - e2) / (e1 + e2);
  }

  function matvec(weight, input, bias) {
    const out = new Array(weight.length);
    for (let i = 0; i < weight.length; i++) {
      const row = weight[i];
      let sum = bias ? bias[i] : 0;
      for (let j = 0; j < row.length; j++) {
        sum += row[j] * input[j];
      }
      out[i] = sum;
    }
    return out;
  }

  function argmax(values) {
    let bestIdx = 0;
    let bestVal = values[0];
    for (let i = 1; i < values.length; i++) {
      if (values[i] > bestVal) {
        bestVal = values[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function buildObs(players, world, selfIndex, stepCount) {
    const me = players[selfIndex];
    const other = players[1 - selfIndex];
    const width = Math.max((world.right ?? 1) - (world.left ?? 0), 1e-6);
    const height = Math.max((world.ceiling ?? 1) - (world.floorLevel ?? 0), 1e-6);

    return [
      me.x / width,
      me.y / height,
      me.vx / 20,
      me.vy / 20,
      other.x / width,
      other.y / height,
      other.vx / 20,
      other.vy / 20,
      (other.x - me.x) / width,
      (other.y - me.y) / height,
      (other.vx - me.vx) / 20,
      (other.vy - me.vy) / 20,
      me.touchingGround ? 1 : 0,
      other.touchingGround ? 1 : 0,
      Math.min(stepCount / 900, 1),
      1,
    ];
  }

  function createPolicy(stateDict) {
    const w1 = stateDict["body.0.weight"];
    const b1 = stateDict["body.0.bias"];
    const w2 = stateDict["body.2.weight"];
    const b2 = stateDict["body.2.bias"];
    const wpi = stateDict["pi.weight"];
    const bpi = stateDict["pi.bias"];

    return {
      act(obs) {
        const h1raw = matvec(w1, obs, b1);
        const h1 = h1raw.map(tanh);
        const h2raw = matvec(w2, h1, b2);
        const h2 = h2raw.map(tanh);
        const logits = matvec(wpi, h2, bpi);
        return argmax(logits);
      },
    };
  }

  async function loadPolicy(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load policy JSON: ${url}`);
    }
    const payload = await response.json();
    return createPolicy(payload.state_dict);
  }

  function createIntentProvider(options) {
    const role = options.role;
    const policy = options.policy;
    let stepCount = 0;

    return {
      nextIntent(player, players, currentMap) {
        stepCount += 1;
        const selfIndex = role === "catcher" ? 0 : 1;
        const world = currentMap.world;
        const obs = buildObs(players, world, selfIndex, stepCount);
        const action = policy.act(obs);
        return ACTION_TO_INTENT[action] ?? ACTION_TO_INTENT[0];
      },
      reset() {
        stepCount = 0;
      },
    };
  }

  global.BonkRLPolicy = {
    loadPolicy,
    createIntentProvider,
  };
})(window);
