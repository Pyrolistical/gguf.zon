const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const gguf = @import("gguf.zig");
const Serializer = std.zon.Serializer;

const ZonValue = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    u64: u64,
    i64: i64,
    f32: f32,
    f64: f64,
    bool_: bool,
    string: []const u8,

    fn deinit(self: ZonValue, gpa: Allocator) void {
        switch (self) {
            .string => |s| gpa.free(s),
            else => {},
        }
    }

    fn serialize(self: ZonValue, s: *Serializer) !void {
        const vo: std.zon.Serializer.ValueOptions = .{};
        switch (self) {
            .u8 => |v| try s.int(v),
            .i8 => |v| try s.int(v),
            .u16 => |v| try s.int(v),
            .i16 => |v| try s.int(v),
            .u32 => |v| try s.int(v),
            .i32 => |v| try s.int(v),
            .u64 => |v| try s.int(v),
            .i64 => |v| try s.int(v),
            .f32 => |v| try s.float(v),
            .f64 => |v| try s.float(v),
            .bool_ => |v| try s.value(v, vo),
            .string => |v| try s.string(v),
        }
    }
};

const MetaTag = enum { scalar, array };

const MetaValue = union(MetaTag) {
    scalar: ZonValue,
    array: []ZonValue,
};

fn deinitMetaValue(gpa: Allocator, mv: *MetaValue) void {
    switch (mv.*) {
        .scalar => |*v| v.deinit(gpa),
        .array => |arr| {
            for (arr) |*v| v.deinit(gpa);
            gpa.free(arr);
        },
    }
}

const MetadataEntry = struct {
    key: []u8,
    value: ?*MetaValue,
};

fn readZonValue(
    r: *Io.Reader,
    gpa: Allocator,
    vt: gguf.ValueType,
) !ZonValue {
    switch (vt) {
        .uint8 => return .{ .u8 = try r.takeInt(u8, .little) },
        .int8 => return .{ .i8 = try r.takeInt(i8, .little) },
        .uint16 => return .{ .u16 = try r.takeInt(u16, .little) },
        .int16 => return .{ .i16 = try r.takeInt(i16, .little) },
        .uint32 => return .{ .u32 = try r.takeInt(u32, .little) },
        .int32 => return .{ .i32 = try r.takeInt(i32, .little) },
        .uint64 => return .{ .u64 = try r.takeInt(u64, .little) },
        .int64 => return .{ .i64 = try r.takeInt(i64, .little) },
        .float32 => return .{ .f32 = @as(f32, @bitCast(try r.takeInt(u32, .little))) },
        .float64 => return .{ .f64 = @as(f64, @bitCast(try r.takeInt(u64, .little))) },
        .bool_ => return .{ .bool_ = (try r.takeInt(u8, .little)) != 0 },
        .string => return .{ .string = try gguf.readGgufString(r, gpa) },
        .array => unreachable,
        _ => return error.UnknownValueType,
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var args = init.minimal.args.iterate();
    _ = args.skip();
    var path_opt: ?[]const u8 = null;
    while (args.next()) |a| {
        if (path_opt == null) {
            path_opt = a;
        } else {
            try printUsage(io);
            return error.UnexpectedArgument;
        }
    }
    const path = path_opt orelse {
        try printUsage(io);
        return error.MissingArgument;
    };

    try runZon(io, gpa, path);
}

fn printUsage(io: Io) !void {
    var ebuf: [128]u8 = undefined;
    var ew = Io.File.stderr().writer(io, &ebuf);
    try ew.interface.writeAll(
        \\usage: gguf.zon <path-to-file.gguf>
        \\
    );
    try ew.interface.flush();
}

fn runZon(io: Io, gpa: Allocator, path: []const u8) !void {
    const file = try Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const file_size: u64 = stat.size;

    var rbuf: [64 * 1024]u8 = undefined;
    var fr = file.reader(io, &rbuf);
    const r = &fr.interface;

    const magic = try r.takeArray(4);
    if (!std.mem.eql(u8, magic, &gguf.MAGIC)) return error.NotAGgufFile;
    const version = try r.takeInt(u32, .little);
    const tensor_count = try r.takeInt(u64, .little);
    const kv_count = try r.takeInt(u64, .little);

    const ARRAY_INLINE_MAX: u64 = 64;

    var meta_list: std.ArrayListUnmanaged(MetadataEntry) = .{ .items = &.{}, .capacity = 0 };
    defer {
        for (meta_list.items) |e| {
            gpa.free(e.key);
            if (e.value) |mv| {
                deinitMetaValue(gpa, mv);
                gpa.destroy(mv);
            }
        }
        meta_list.deinit(gpa);
    }

    var alignment: u64 = gguf.DEFAULT_ALIGNMENT;
    var vocab_size: u64 = 0;

    var i: u64 = 0;
    while (i < kv_count) : (i += 1) {
        const key = try gguf.readGgufString(r, gpa);
        const vt: gguf.ValueType = @enumFromInt(try r.takeInt(u32, .little));

        if (std.mem.eql(u8, key, "general.alignment")) {
            switch (vt) {
                .uint32 => alignment = try r.takeInt(u32, .little),
                .uint64 => alignment = try r.takeInt(u64, .little),
                .int32 => {
                    const v = try r.takeInt(i32, .little);
                    if (v > 0) alignment = @intCast(v);
                },
                .int64 => {
                    const v = try r.takeInt(i64, .little);
                    if (v > 0) alignment = @intCast(v);
                },
                else => try gguf.skipValue(r, vt),
            }
            try meta_list.append(gpa, .{ .key = key, .value = null });
            continue;
        }

        if (vt == .array) {
            const elem_t: gguf.ValueType = @enumFromInt(try r.takeInt(u32, .little));
            const len = try r.takeInt(u64, .little);

            if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) vocab_size = len;

            if (len > ARRAY_INLINE_MAX) {
                var k: u64 = 0;
                while (k < len) : (k += 1) try gguf.skipValue(r, elem_t);
                try meta_list.append(gpa, .{ .key = key, .value = null });
                continue;
            }

            var arr: []ZonValue = try gpa.alloc(ZonValue, len);
            var k: u64 = 0;
            while (k < len) : (k += 1) {
                arr[k] = try readZonValue(r, gpa, elem_t);
            }
            const mv = try gpa.create(MetaValue);
            mv.* = .{ .array = arr };
            try meta_list.append(gpa, .{ .key = key, .value = mv });
            continue;
        }

        const zv = try readZonValue(r, gpa, vt);
        const mv = try gpa.create(MetaValue);
        mv.* = .{ .scalar = zv };
        try meta_list.append(gpa, .{ .key = key, .value = mv });
    }

    var t: u64 = 0;
    while (t < tensor_count) : (t += 1) {
        const name = try gguf.readGgufString(r, gpa);
        defer gpa.free(name);

        const n_dims = try r.takeInt(u32, .little);
        var d: u32 = 0;
        while (d < n_dims) : (d += 1) _ = try r.takeInt(u64, .little);
        _ = try r.takeInt(u32, .little);
        _ = try r.takeInt(u64, .little);
    }

    const cur = fr.logicalPos();

    const data_start = std.mem.alignForward(u64, cur, alignment);

    var wbuf: [256 * 1024]u8 = undefined;
    var ow = Io.File.stdout().writer(io, &wbuf);
    const w = &ow.interface;
    defer w.flush() catch {};

    var s: Serializer = .{ .writer = w };
    const vo: std.zon.Serializer.ValueOptions = .{};

    var root = try s.beginStruct(.{});

    try root.field("gguf_version", version, vo);
    try root.field("file_size", file_size, vo);

    try root.field("weights_offset", data_start, vo);
    try root.field("weights_end", file_size, vo);

    var meta = try root.beginStructField("metadata", .{});
    for (meta_list.items) |entry| {
        if (std.mem.eql(u8, entry.key, "general.alignment")) {
            try meta.field(entry.key, alignment, vo);
            continue;
        }

        const mv = entry.value orelse continue;

        try meta.fieldPrefix(entry.key);
        switch (mv.*) {
            .scalar => |zv| try zv.serialize(&s),
            .array => |arr| {
                var tuple = try s.beginTuple(.{ .whitespace_style = .{ .fields = arr.len } });
                for (arr) |zv| {
                    try tuple.fieldPrefix();
                    try zv.serialize(&s);
                }
                try tuple.end();
            },
        }
    }
    try meta.end();

    if (vocab_size > 0) try root.field("vocab_size", vocab_size, vo);

    try root.field("tensor_count", tensor_count, vo);
    try root.field("alignment", alignment, vo);
    try root.field("data_start", data_start, vo);

    try root.end();
    try w.writeByte('\n');
}
