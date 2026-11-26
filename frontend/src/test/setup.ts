import "@testing-library/jest-dom/vitest";

// Stub canvas getContext for jsdom unit tests
HTMLCanvasElement.prototype.getContext = (() =>
  ({
    setTransform: () => {},
    clearRect: () => {},
    fillRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    arc: () => {},
    fill: () => {},
    fillText: () => {},
    createRadialGradient: () => ({ addColorStop: () => {} }),
  }) as unknown) as typeof HTMLCanvasElement.prototype.getContext;
