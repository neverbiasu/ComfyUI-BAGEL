import { app } from "../../scripts/app.js";

const NATIVE_T2I = "BAGELTextToImage";

/**
 * Migrate the positional widgets in T2I workflows saved before v1.1.0.
 *
 * ComfyUI serializes widget values as an array.  The native node used to
 * expose `cfg_img_scale` between cfg_text_scale and num_timesteps; it is not
 * a user setting in BAGEL's T2I app and was removed.  Without this migration,
 * every following old widget is read as the wrong new input (notably a zero
 * can reach max_think_tokens).
 */
function migrateNativeT2IWidgets(values) {
    if (!Array.isArray(values) || values.length < 9) return values;

    const oldCfgImgScale = values[4];
    const oldNumTimesteps = values[5];
    const isPreV110 =
        typeof oldCfgImgScale === "number" && oldCfgImgScale >= 1 &&
        typeof oldNumTimesteps === "number" && oldNumTimesteps >= 10;
    if (!isPreV110) return values;

    const hasAdvancedValues = typeof values[8] === "boolean";
    const seedControl = typeof values.at(-1) === "string" ? values.at(-1) : undefined;
    const showThinking = hasAdvancedValues ? values[8] : false;
    const cfgInterval = hasAdvancedValues ? values[9] : 0.4;
    const cfgRenormMin = hasAdvancedValues ? values[10] : 0.0;
    const cfgRenormType = hasAdvancedValues ? values[11] : "global";
    const textTemperature = hasAdvancedValues ? values[12] : 0.3;

    const migrated = [
        values[0], // prompt
        values[1], // width
        values[2], // height
        values[3], // cfg_text_scale
        cfgInterval,
        values[6], // timestep_shift
        values[5], // num_timesteps
        cfgRenormMin,
        cfgRenormType,
        showThinking,
        1024, // max_think_tokens: official app default
        false, // do_sample: official app default
        textTemperature,
        values[7], // seed
    ];
    if (seedControl !== undefined) migrated.push(seedControl);
    return migrated;
}

app.registerExtension({
    name: "ComfyUI.BAGEL.WorkflowMigration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NATIVE_T2I) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure(info) {
            if (info?.widgets_values) {
                info.widgets_values = migrateNativeT2IWidgets(info.widgets_values);
            }
            return originalOnConfigure?.apply(this, arguments);
        };
    },
});
