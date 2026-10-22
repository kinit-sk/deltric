// Browser regression checks, run against a local server with Playwright CLI:
// playwright-cli run-code "$(cat deltric_lab/test_browser.js)"
async (page) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // Test navigation must not overwrite the investigator's saved workspace.
  await page.route("**/api/workspace", async (route) => {
    if (route.request().method() === "POST")
      await route.fulfill({ json: { ok: true } });
    else await route.continue();
  });
  await page.reload();
  await page.evaluate(() => { navigator.sendBeacon = () => true; });
  await page.getByRole("button", { name: /Gaia · ngc 6705/ }).click();
  await page.waitForFunction(() => S.name === "gaia_ngc_6705" && S.data && !S.busy);
  await page.getByRole("spinbutton", { name: "Point ID", exact: true }).fill("1269");
  await page.getByRole("button", { name: "Find →", exact: true }).click();
  await page.getByRole("heading", { name: "Point #1269", exact: true }).waitFor();
  const results = await page.evaluate(async () => {
    const passed = [];
    const check = (ok, name) => {
      if (!ok) throw new Error(name);
      passed.push(name);
    };
    check(document.querySelector(".point-outcome .tag.green")?.textContent === "CORRECTLY UNASSIGNED",
      "Screenshot point #1269 is correctly unassigned and green");
    check(!document.querySelector(".point-heading .tag").classList.contains("red"),
      "Ground-truth badge is neutral");
    check(pointOutcome(0, -1).tone === "amber" && pointOutcome(0, 0).tone === "neutral",
      "Normal assignment does not imply correct destination cluster");

    $("pipeline-mode").value = "reassignment";
    $("graph").checked = false;
    $("frontier").checked = false;
    $("tree").checked = true;
    $("paths").checked = true;
    $("cluster-focus").value = "all";
    $("point-filter").value = "all";
    S.cameras = {};
    plots[0].fit(false);
    const ctx = $("canvas-a").getContext("2d");
    const strokes = [];
    let points = [];
    const originals = Object.fromEntries(["beginPath", "moveTo", "lineTo", "stroke"].map((key) => [key, ctx[key]]));
    ctx.beginPath = function (...args) { points = []; return originals.beginPath.apply(this, args); };
    ctx.moveTo = function (...args) { points.push(args); return originals.moveTo.apply(this, args); };
    ctx.lineTo = function (...args) { points.push(args); return originals.lineTo.apply(this, args); };
    ctx.stroke = function (...args) {
      if (points.length === 2) strokes.push({ width: this.lineWidth, points: points.slice() });
      return originals.stroke.apply(this, args);
    };
    const capture = (view) => {
      $("view-a").value = view;
      strokes.length = 0;
      renderPlots();
      return strokes.slice();
    };
    const matches = (actual, claims, width) => {
      const lines = actual.filter((s) => Math.abs(s.width - width) < 0.001);
      return lines.length === claims.length && claims.every((c, i) =>
        [c[4], c[0]].every((node, j) => plots[0].screen(plots[0].xy[node]).every((v, k) =>
          Math.abs(v - lines[i].points[j][k]) < 0.001)));
    };
    try {
      setFrame(0, { persist: false, inspect: false });
      check(matches(capture("current"), [], 1.8), "No claim edges before replay starts");
      setFrame(5, { persist: false, inspect: false });
      check(matches(capture("current"), S.data.claims.slice(0, 5), 1.8),
        "Replay draws exactly five production parent-to-target edges");
      check(matches(capture("before"), [], 1.8), "Before view has no claim edges");
      check(matches(capture("truth"), [], 1.8), "Ground truth has no claim edges");
      check(matches(capture("final"), S.data.claims, 1.8), "Final view draws the full active trace");
      check($("stage-a").textContent === `${S.data.claims.length} claims`,
        "Final caption shows its full trace count while replay is at claim five");
      check(matches(capture("default"), S.data.recorded_claims, 1.8), "Recorded default draws its own full trace");

      const harmIndex = S.data.claims.findIndex((c) => S.data.y[c[0]] < 0);
      const harm = S.data.claims[harmIndex];
      $("view-a").value = "current";
      await inspectPoint(harm[0]);
      setFrame(harmIndex, { persist: false, inspect: false });
      check(document.querySelector(".point-outcome .tag.green") &&
        document.querySelector(".explanation").textContent.includes("Still unassigned"),
        "Scrubbing before the anomaly claim updates status immediately");
      check(document.querySelectorAll(".hop-card.future").length > 0,
        "Future path hops are visibly marked");
      setFrame(harmIndex + 1, { persist: false, inspect: false });
      check(document.querySelector(".point-outcome .tag.red")?.textContent === "ANOMALY ASSIGNED",
        "An absorbed anomaly turns red at its exact claim");
      check(matches(capture("before"), [], 2.8) && matches(capture("truth"), [], 2.8),
        "Selected claim paths are absent in before and truth views");

      const multi = S.data.claims.find((c) => winningPath(c[0], S.data.claims).length > 2);
      check(!!multi, "Real Gaia case contains a multi-hop winning path");
      await inspectPoint(multi[0]);
      setFrame(S.data.claims.length, { persist: false, inspect: false });
      const selected = winningPath(multi[0], S.data.claims);
      const selectedClaims = selected.slice(1).map((n) => S.data.claims.find((c) => c[0] === n));
      check(matches(capture("current"), selectedClaims, 2.8),
        "Selected multi-hop canvas path follows every production parent");
      $("paths").checked = false;
      check(matches(capture("current"), [], 2.8), "Selected-path switch hides the path");
      $("paths").checked = true;
      $("point-filter").value = "anomalies";
      check(matches(capture("current"), S.data.claims.filter((c) => S.data.y[c[0]] < 0), 1.8),
        "Claim-path layer respects the anomaly point filter");
      $("point-filter").value = "all";
      $("cluster-focus").value = String(multi[1]);
      check(matches(capture("current"), S.data.claims.filter((c) => c[1] === multi[1]), 1.8),
        "Claim-path layer respects source-cluster focus");
      $("cluster-focus").value = "all";
      $("projection").value = "pca";
      plots[0].fit(false);
      check(matches(capture("final"), S.data.claims, 1.8), "Paths use the selected PCA projection");
      $("projection").value = "graph";

      const active = S.data;
      S.data = { ...active, claims: [], final: active.base.slice() };
      setFrame(0, { persist: false, inspect: false });
      check(matches(capture("final"), [], 1.8) &&
        matches(capture("default"), active.recorded_claims, 1.8),
        "An empty experiment trace preserves the recorded comparison paths");
      $("cluster-focus").value = String(multi[1]);
      check(matches(capture("default"), active.recorded_claims.filter((c) => c[1] === multi[1]), 1.8),
        "Recorded comparison cluster focus works when the active run made no claims");
      $("cluster-focus").value = "all";
      S.data = active;

      $("pipeline-mode").value = "full";
      setFrame(0, { persist: false, inspect: false });
      check(matches(capture("current"), [], 1.8) && matches(capture("current"), [], 2.8),
        "Upstream stages show no later reassignment paths");
      setFrame(total(), { persist: false, inspect: false });
      await inspectPoint(1269);
      check(matches(capture("current"), [], 2.8), "Unclaimed point never inherits the previous selection's path");
    } finally {
      clearTimeout(saveTimer);
      Object.assign(ctx, originals);
      $("view-a").value = "current";
      $("color-a").value = "cluster";
      $("graph").checked = true;
      $("frontier").checked = true;
      renderPlots();
    }
    window.deltricTestResults = passed;
    return passed;
  });
  if (errors.length) throw new Error(errors.join("\n"));
  return { passed: results.length, checks: results };
}
