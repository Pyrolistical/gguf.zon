const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const exe = b.addExecutable(.{
        .name = "gguf.zon",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run gguf.zon");
    const cmd = b.addRunArtifact(exe);
    cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| cmd.addArgs(args);
    run_step.dependOn(&cmd.step);
}
