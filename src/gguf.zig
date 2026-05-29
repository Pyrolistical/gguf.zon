const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const MAGIC: [4]u8 = .{ 'G', 'G', 'U', 'F' };
pub const DEFAULT_ALIGNMENT: u64 = 32;

pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    _,

    pub fn name(t: GgmlType) []const u8 {
        return switch (t) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q8_0 => "Q8_0",
            .q8_1 => "Q8_1",
            .q2_k => "Q2_K",
            .q3_k => "Q3_K",
            .q4_k => "Q4_K",
            .q5_k => "Q5_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
            .iq2_xxs => "IQ2_XXS",
            .iq2_xs => "IQ2_XS",
            .iq3_xxs => "IQ3_XXS",
            .iq1_s => "IQ1_S",
            .iq4_nl => "IQ4_NL",
            .iq3_s => "IQ3_S",
            .iq2_s => "IQ2_S",
            .iq4_xs => "IQ4_XS",
            .i8 => "I8",
            .i16 => "I16",
            .i32 => "I32",
            .i64 => "I64",
            .f64 => "F64",
            .iq1_m => "IQ1_M",
            .bf16 => "BF16",
            .tq1_0 => "TQ1_0",
            .tq2_0 => "TQ2_0",
            _ => "UNKNOWN",
        };
    }

    pub fn blockBytes(t: GgmlType) u32 {
        return switch (t) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .f64, .i64 => 8,
            .i8 => 1,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq2_s => 82,
            .iq3_xxs => 98,
            .iq3_s => 110,
            .iq1_s => 50,
            .iq1_m => 56,
            .iq4_nl => 18,
            .iq4_xs => 136,
            .tq1_0 => 54,
            .tq2_0 => 66,
            _ => 0,
        };
    }

    pub fn blockElems(t: GgmlType) u32 {
        return switch (t) {
            .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_xs, .tq1_0, .tq2_0 => 256,
            _ => 0,
        };
    }
};

pub const Header = struct {
    version: u32,
    tensor_count: u64,
    kv_count: u64,
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    type: GgmlType,
    offset: u64,

    pub fn elemCount(t: TensorInfo) u64 {
        var n: u64 = 1;
        var i: u32 = 0;
        while (i < t.n_dims) : (i += 1) n *= t.dims[i];
        return n;
    }

    pub fn byteSize(t: TensorInfo) u64 {
        const n = t.elemCount();
        const eb = t.type.blockElems();
        const bb = t.type.blockBytes();
        if (eb == 0 or bb == 0) return 0;
        return (n / eb) * bb;
    }
};

pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    header: Header,
    alignment: u64,
    data_start: u64,
    infos: []TensorInfo,
    by_name: std.StringHashMapUnmanaged(u32),

    rope_freq_base: f32,
    rms_norm_eps: f32,
    rope_dim_count: u32,

    pub fn deinit(self: *Index) void {
        self.arena.deinit();
    }

    pub fn get(self: *const Index, name: []const u8) ?*const TensorInfo {
        const i = self.by_name.get(name) orelse return null;
        return &self.infos[i];
    }
};

pub fn readGgufString(r: *Io.Reader, gpa: Allocator) ![]u8 {
    const len = try r.takeInt(u64, .little);
    return r.readAlloc(gpa, std.math.cast(usize, len) orelse return error.StringTooLarge);
}

pub fn skipValue(r: *Io.Reader, vt: ValueType) !void {
    switch (vt) {
        .uint8, .int8, .bool_ => try r.discardAll(1),
        .uint16, .int16 => try r.discardAll(2),
        .uint32, .int32, .float32 => try r.discardAll(4),
        .uint64, .int64, .float64 => try r.discardAll(8),
        .string => {
            const len = try r.takeInt(u64, .little);
            try r.discardAll64(len);
        },
        .array => {
            const elem_t: ValueType = @enumFromInt(try r.takeInt(u32, .little));
            const len = try r.takeInt(u64, .little);
            var i: u64 = 0;
            while (i < len) : (i += 1) try skipValue(r, elem_t);
        },
        _ => return error.UnknownValueType,
    }
}

pub const KvHandler = struct {
    ctx: *anyopaque,
    onKv: *const fn (ctx: *anyopaque, key: []const u8, vt: ValueType, r: *Io.Reader) anyerror!void,
};

pub fn streamHeaderAndMeta(r: *Io.Reader, gpa: Allocator, kv: ?KvHandler) !Header {
    const magic = try r.takeArray(4);
    if (!std.mem.eql(u8, magic, &MAGIC)) return error.NotAGgufFile;
    const version = try r.takeInt(u32, .little);
    const tc = try r.takeInt(u64, .little);
    const kc = try r.takeInt(u64, .little);

    var i: u64 = 0;
    while (i < kc) : (i += 1) {
        const key = try readGgufString(r, gpa);
        defer gpa.free(key);
        const vt: ValueType = @enumFromInt(try r.takeInt(u32, .little));
        if (kv) |h| {
            try h.onKv(h.ctx, key, vt, r);
        } else {
            try skipValue(r, vt);
        }
    }

    return .{ .version = version, .tensor_count = tc, .kv_count = kc };
}

pub fn readTensorInfos(
    r: *Io.Reader,
    arena: Allocator,
    out: []TensorInfo,
) !void {
    for (out) |*ti| {
        const name_len = try r.takeInt(u64, .little);
        const nl: usize = std.math.cast(usize, name_len) orelse return error.StringTooLarge;
        const name = try arena.alloc(u8, nl);
        try r.readSliceAll(name);

        const n_dims = try r.takeInt(u32, .little);
        if (n_dims > 4) return error.TooManyDims;
        var dims: [4]u64 = .{ 0, 0, 0, 0 };
        var d: u32 = 0;
        while (d < n_dims) : (d += 1) dims[d] = try r.takeInt(u64, .little);

        const type_raw = try r.takeInt(u32, .little);
        const offset = try r.takeInt(u64, .little);

        ti.* = .{
            .name = name,
            .n_dims = n_dims,
            .dims = dims,
            .type = @enumFromInt(type_raw),
            .offset = offset,
        };
    }
}

pub fn loadIndex(
    file_reader: *std.Io.File.Reader,
    gpa: Allocator,
) !Index {
    const r = &file_reader.interface;

    var idx_arena = std.heap.ArenaAllocator.init(gpa);
    errdefer idx_arena.deinit();
    const aa = idx_arena.allocator();

    var alignment: u64 = DEFAULT_ALIGNMENT;
    var rope_freq_base: f32 = 1.0e7;
    var rms_norm_eps: f32 = 1.0e-6;
    var rope_dim_count: u32 = 64;
    const Probe = struct {
        alignment_out: *u64,
        rope_freq_base_out: *f32,
        rms_norm_eps_out: *f32,
        rope_dim_count_out: *u32,
        gpa: Allocator,

        fn cb(ctx: *anyopaque, key: []const u8, vt: ValueType, rr: *Io.Reader) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (std.mem.eql(u8, key, "general.alignment")) {
                switch (vt) {
                    .uint32 => self.alignment_out.* = try rr.takeInt(u32, .little),
                    .uint64 => self.alignment_out.* = try rr.takeInt(u64, .little),
                    .int32 => {
                        const v = try rr.takeInt(i32, .little);
                        if (v > 0) self.alignment_out.* = @intCast(v);
                    },
                    .int64 => {
                        const v = try rr.takeInt(i64, .little);
                        if (v > 0) self.alignment_out.* = @intCast(v);
                    },
                    else => try skipValue(rr, vt),
                }
            } else if (std.mem.eql(u8, key, "qwen35.rope.freq_base") and vt == .float32) {
                self.rope_freq_base_out.* = @bitCast(try rr.takeInt(u32, .little));
            } else if (std.mem.eql(u8, key, "qwen35.attention.layer_norm_rms_epsilon") and vt == .float32) {
                self.rms_norm_eps_out.* = @bitCast(try rr.takeInt(u32, .little));
            } else if (std.mem.eql(u8, key, "qwen35.rope.dimension_count") and vt == .uint32) {
                self.rope_dim_count_out.* = try rr.takeInt(u32, .little);
            } else {
                try skipValue(rr, vt);
            }
        }
    };
    var probe: Probe = .{
        .alignment_out = &alignment,
        .rope_freq_base_out = &rope_freq_base,
        .rms_norm_eps_out = &rms_norm_eps,
        .rope_dim_count_out = &rope_dim_count,
        .gpa = gpa,
    };
    const header = try streamHeaderAndMeta(r, gpa, .{
        .ctx = @ptrCast(&probe),
        .onKv = Probe.cb,
    });

    const tc: usize = std.math.cast(usize, header.tensor_count) orelse
        return error.TooManyTensors;
    const infos = try aa.alloc(TensorInfo, tc);
    try readTensorInfos(r, aa, infos);

    const cur = file_reader.logicalPos();
    const data_start = std.mem.alignForward(u64, cur, alignment);

    var by_name: std.StringHashMapUnmanaged(u32) = .empty;
    try by_name.ensureTotalCapacity(aa, @intCast(tc));
    for (infos, 0..) |ti, i| {
        try by_name.put(aa, ti.name, @intCast(i));
    }

    return .{
        .arena = idx_arena,
        .header = header,
        .alignment = alignment,
        .data_start = data_start,
        .infos = infos,
        .by_name = by_name,
        .rope_freq_base = rope_freq_base,
        .rms_norm_eps = rms_norm_eps,
        .rope_dim_count = rope_dim_count,
    };
}

pub fn readWeights(
    file: Io.File,
    io: Io,
    data_start: u64,
    dst: []u8,
) !usize {
    const chunk: usize = 4 * 1024 * 1024;
    var off: usize = 0;
    while (off < dst.len) {
        const want = @min(chunk, dst.len - off);
        const got = try file.readPositionalAll(io, dst[off .. off + want], data_start + off);
        if (got == 0) break;
        off += got;
        if (got < want) break;
    }
    return off;
}
