export const BACKEND_URL = "https://2ca802009300.ngrok-free.app";

export const DEVICE_SPECS = {
  // UPDATE THESE FOR YOUR SPECIFIC PHONE
  // Example values are for iPhone 13 Pro in landscape-right orientation
  screen: {
    widthCm: 13.2,      // Physical width in landscape (cm)
    heightCm: 6.1,      // Physical height in landscape (cm)
    widthPx: 2532,      // Resolution width in landscape (px)
    heightPx: 1170,     // Resolution height in landscape (px)
  },
  camera: {
    // Front camera position in landscape mode
    positionCm: { x: 1.5, y: 3.05 },    // Position from top-left in cm
    positionPx: { x: 287, y: 585 },      // Position from top-left in px
  },
};

