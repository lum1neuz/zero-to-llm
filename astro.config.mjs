import { defineConfig } from "astro/config";
import react from "@astrojs/react";

export default defineConfig({
  integrations: [react()],
  output: "static",
  trailingSlash: "ignore",
  build: { format: "directory" },
  devToolbar: { enabled: false },
});
