#include "testing.h"

#include "../tools/server/server-common.h"
#include "../tools/server/server-decision.h"
#include "../tools/server/server-adaptive-draft.h"
#include "../tools/server/server-schema.h"
#include "speculative.h"

#include <cstdlib>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// unit tests for server internal components, no model required
// NOTE: end-to-end tests live in tools/server/tests
// see skills/code-review/SKILL.md for more

//
// server_tokens (text only, has_mtmd = false)
//

static void test_server_tokens_basic(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    t.assert_equal("size", (size_t) 3, st.size());
    t.assert_true("not empty", !st.empty());
    t.assert_equal("operator[]", 2, st[1]);

    st.push_back(4);
    t.assert_equal("push_back", (size_t) 4, st.size());
    t.assert_true("get_tokens", st.get_tokens() == llama_tokens({1, 2, 3, 4}));

    st.insert({5, 6});
    t.assert_true("insert appends", st.get_tokens() == llama_tokens({1, 2, 3, 4, 5, 6}));

    st.set_token(0, 9);
    t.assert_equal("set_token", 9, st[0]);

    bool thrown = false;
    try {
        st.push_back(LLAMA_TOKEN_NULL);
    } catch (const std::exception &) {
        thrown = true;
    }
    t.assert_true("push_back(LLAMA_TOKEN_NULL) throws", thrown);
    t.assert_equal("size unchanged after throw", (size_t) 6, st.size());

    st.clear();
    t.assert_true("clear", st.empty());
}

static void test_server_tokens_pos(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    t.assert_equal("pos_next()", (llama_pos) 5, st.pos_next());
    t.assert_equal("pos_next(0)", (llama_pos) 0, st.pos_next(0));
    t.assert_equal("pos_next(3)", (llama_pos) 3, st.pos_next(3));

    t.assert_equal("size_up_to_pos(0)", (size_t) 0, st.size_up_to_pos(0));
    t.assert_equal("size_up_to_pos(3)", (size_t) 3, st.size_up_to_pos(3));
    t.assert_equal("size_up_to_pos clamps", (size_t) 5, st.size_up_to_pos(100));
}

static void test_server_tokens_keep_first(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3, 4, 5}, false);
    st.keep_first(5);
    t.assert_equal("keep all", (size_t) 5, st.size());
    st.keep_first(2);
    t.assert_true("keep 2", st.get_tokens() == llama_tokens({1, 2}));
    st.keep_first(0);
    t.assert_true("keep 0", st.empty());
}

static void test_server_tokens_common_prefix(testing & t) {
    server_tokens a(llama_tokens{1, 2, 3, 4}, false);
    server_tokens empty(llama_tokens{}, false);

    t.assert_equal("both empty",  (size_t) 0, empty.get_common_prefix(empty));
    t.assert_equal("one empty",   (size_t) 0, a.get_common_prefix(empty));
    t.assert_equal("identical",   (size_t) 4, a.get_common_prefix(a));
    t.assert_equal("diverge at 2", (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 9, 4}, false)));
    t.assert_equal("diverge at 0", (size_t) 0, a.get_common_prefix(server_tokens(llama_tokens{9, 2, 3, 4}, false)));
    t.assert_equal("b is prefix",  (size_t) 2, a.get_common_prefix(server_tokens(llama_tokens{1, 2}, false)));
    t.assert_equal("a is prefix",  (size_t) 4, a.get_common_prefix(server_tokens(llama_tokens{1, 2, 3, 4, 5, 6}, false)));
}

static void test_server_tokens_serialize(testing & t) {
    // same packing as slot save/restore in server-context.cpp
    auto pack = [](const std::vector<char> & data) {
        llama_tokens packed(data.size() / sizeof(llama_token));
        memcpy(packed.data(), data.data(), data.size());
        return packed;
    };

    t.test("roundtrip", [&](testing & t) {
        server_tokens st(llama_tokens{10, 20, 30}, false);
        std::vector<char> data = st.serialize();
        t.assert_equal("aligned to llama_token", (size_t) 0, data.size() % sizeof(llama_token));

        server_tokens restored = server_tokens::deserialize(pack(data), false);
        t.assert_true("tokens", restored.get_tokens() == st.get_tokens());
    });

    t.test("roundtrip empty", [&](testing & t) {
        server_tokens st(llama_tokens{}, false);
        server_tokens restored = server_tokens::deserialize(pack(st.serialize()), false);
        t.assert_true("empty", restored.empty());
    });

    t.test("legacy plain token list", [&](testing & t) {
        server_tokens restored = server_tokens::deserialize(llama_tokens{7, 8, 9}, false);
        t.assert_true("tokens", restored.get_tokens() == llama_tokens({7, 8, 9}));
    });

    auto expect_throw = [](testing & t, const std::string & msg, const llama_tokens & packed) {
        bool thrown = false;
        try {
            server_tokens::deserialize(packed, false);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true(msg, thrown);
    };

    t.test("corrupted", [&](testing & t) {
        llama_tokens good = pack(server_tokens(llama_tokens{10, 20, 30}, false).serialize());

        llama_tokens bad_version = good;
        bad_version[1] = 999;
        expect_throw(t, "bad version", bad_version);

        llama_tokens truncated = good;
        truncated.pop_back();
        expect_throw(t, "truncated", truncated);

        llama_tokens trailing = good;
        trailing.push_back(42);
        expect_throw(t, "trailing data", trailing);

        llama_tokens huge_count = good;
        huge_count[2] = 0x7fffffff;
        expect_throw(t, "huge token count", huge_count);

        expect_throw(t, "marker only", llama_tokens{LLAMA_TOKEN_NULL});
    });
}

static void test_server_tokens_clone(testing & t) {
    server_tokens st(llama_tokens{1, 2, 3}, false);
    server_tokens cl = st.clone();
    cl.push_back(4);
    t.assert_equal("original untouched", (size_t) 3, st.size());
    t.assert_equal("clone grew", (size_t) 4, cl.size());
    t.assert_true("get_text_tokens", cl.get_text_tokens() == llama_tokens({1, 2, 3, 4}));
}

static void test_server_tokens(testing & t) {
    t.test("basic",         test_server_tokens_basic);
    t.test("pos",           test_server_tokens_pos);
    t.test("keep_first",    test_server_tokens_keep_first);
    t.test("common_prefix", test_server_tokens_common_prefix);
    t.test("serialize",     test_server_tokens_serialize);
    t.test("clone",         test_server_tokens_clone);
}

//
// string / json helpers
//

static void test_utf8(testing & t) {
    t.test("validate_utf8", [](testing & t) {
        t.assert_equal("empty",            (size_t) 0, validate_utf8(""));
        t.assert_equal("ascii",            (size_t) 3, validate_utf8("abc"));
        t.assert_equal("full 2-byte",      (size_t) 3, validate_utf8("a\xC3\xA9"));
        t.assert_equal("full 3-byte",      (size_t) 4, validate_utf8("a\xE2\x82\xAC"));
        t.assert_equal("full 4-byte",      (size_t) 5, validate_utf8("a\xF0\x9F\x98\x80"));
        t.assert_equal("cut 2-byte (1/2)", (size_t) 1, validate_utf8("a\xC3"));
        t.assert_equal("cut 3-byte (1/3)", (size_t) 1, validate_utf8("a\xE2"));
        t.assert_equal("cut 3-byte (2/3)", (size_t) 1, validate_utf8("a\xE2\x82"));
        t.assert_equal("cut 4-byte (1/4)", (size_t) 1, validate_utf8("a\xF0"));
        t.assert_equal("cut 4-byte (2/4)", (size_t) 1, validate_utf8("a\xF0\x9F"));
        t.assert_equal("cut 4-byte (3/4)", (size_t) 1, validate_utf8("a\xF0\x9F\x98"));
        t.assert_equal("cut only",         (size_t) 0, validate_utf8("\xF0\x9F\x98"));
        t.assert_equal("full then cut",    (size_t) 4, validate_utf8("\xF0\x9F\x98\x80\xC3"));
    });

    t.test("is_valid_utf8", [](testing & t) {
        t.assert_true("empty",          is_valid_utf8(""));
        t.assert_true("ascii",          is_valid_utf8("hello"));
        t.assert_true("multi-byte",     is_valid_utf8("\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80"));
        t.assert_true("truncated",      !is_valid_utf8("\xE2\x82"));
        t.assert_true("bad cont. byte", !is_valid_utf8("\xC3\x41"));
        t.assert_true("lone cont. byte", !is_valid_utf8("\x80"));
        t.assert_true("invalid lead",   !is_valid_utf8("\xFF"));
    });
}

static void test_sse(testing & t) {
    t.test("oai", [](testing & t) {
        t.assert_equal("single", std::string("data: {\"a\":1}\n\n"), format_oai_sse(json{{"a", 1}}));
        t.assert_equal("array",  std::string("data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"),
                format_oai_sse(json::array({json{{"a", 1}}, json{{"b", 2}}})));
        t.assert_equal("invalid utf8 replaced", std::string("data: \"x\xEF\xBF\xBD\"\n\n"), format_oai_sse(json("x\xC3")));
    });

    t.test("oai_resp", [](testing & t) {
        json ev = {{"event", "response.created"}, {"data", {{"id", "r1"}}}};
        t.assert_equal("single", std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\n"), format_oai_resp_sse(ev));
        t.assert_equal("array",  std::string("event: response.created\ndata: {\"id\":\"r1\"}\n\nevent: response.created\ndata: {\"id\":\"r1\"}\n\n"),
                format_oai_resp_sse(json::array({ev, ev})));
    });

    t.test("anthropic", [](testing & t) {
        json ev = {{"event", "message_start"}, {"data", {{"type", "message_start"}}}};
        t.assert_equal("with event", std::string("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"), format_anthropic_sse(ev));
        t.assert_equal("data only",  std::string("data: {\"type\":\"ping\"}\n\n"), format_anthropic_sse(json{{"type", "ping"}}));
    });
}

static void test_json_helpers(testing & t) {
    t.test("json_is_array_of_numbers", [](testing & t) {
        t.assert_true("ints",         json_is_array_of_numbers(json::array({1, 2, 3})));
        t.assert_true("empty array",  json_is_array_of_numbers(json::array()));
        t.assert_true("float",        !json_is_array_of_numbers(json::array({1, 2.5})));
        t.assert_true("string",       !json_is_array_of_numbers(json::array({1, "a"})));
        t.assert_true("not an array", !json_is_array_of_numbers(json(1)));
    });

    t.test("json_is_array_of_mixed_numbers_strings", [](testing & t) {
        t.assert_true("mixed",        json_is_array_of_mixed_numbers_strings(json::array({"a", 1})));
        t.assert_true("numbers only", !json_is_array_of_mixed_numbers_strings(json::array({1, 2})));
        t.assert_true("strings only", !json_is_array_of_mixed_numbers_strings(json::array({"a", "b"})));
        t.assert_true("empty",        !json_is_array_of_mixed_numbers_strings(json::array()));
    });

    t.test("json_is_array_and_contains_numbers", [](testing & t) {
        t.assert_true("contains",     json_is_array_and_contains_numbers(json::array({"a", 1})));
        t.assert_true("strings only", !json_is_array_and_contains_numbers(json::array({"a"})));
        t.assert_true("not an array", !json_is_array_and_contains_numbers(json(1)));
    });

    t.test("json_get_nested_values", [](testing & t) {
        json js = {{"a", {{"b", {{"c", 1}}}}}, {"x", 2}};
        json res = json_get_nested_values({"a/b/c", "x", "a/b", "a/missing", "x/y"}, js);
        t.assert_equal("nested",  1, res.at("a/b/c").get<int>());
        t.assert_equal("top",     2, res.at("x").get<int>());
        t.assert_true("object",   res.at("a/b") == json{{"c", 1}});
        t.assert_true("missing",  !res.contains("a/missing"));
        t.assert_true("not obj",  !res.contains("x/y"));
        t.assert_equal("count", (size_t) 3, res.size());
    });
}

static void test_lora_helpers(testing & t) {
    // only pointer identity is used, never dereferenced
    auto * p0 = reinterpret_cast<llama_adapter_lora *>(0x10);
    auto * p1 = reinterpret_cast<llama_adapter_lora *>(0x20);

    auto make = [](llama_adapter_lora * ptr, float scale) {
        common_adapter_lora_info info;
        info.scale = scale;
        info.ptr   = ptr;
        return info;
    };

    t.test("parse_lora_request", [](testing & t) {
        auto res = parse_lora_request(json::array({
            json{{"id", 0}, {"scale", 0.5}},
            json{{"id", 2}, {"scale", 1.0}},
            json{{"id", 0}, {"scale", 0.25}}, // last one wins
        }));
        t.assert_equal("count", (size_t) 2, res.size());
        t.assert_equal("id 0", 0.25f, res.at(0));
        t.assert_equal("id 2", 1.0f,  res.at(2));

        auto defaults = parse_lora_request(json::array({json::object()}));
        t.assert_equal("default id/scale", 0.0f, defaults.at(-1));
    });

    t.test("are_lora_equal", [&](testing & t) {
        t.assert_true("both empty",   are_lora_equal({}, {}));
        t.assert_true("same",         are_lora_equal({make(p0, 1.0f)}, {make(p0, 1.0f)}));
        t.assert_true("diff scale",   !are_lora_equal({make(p0, 1.0f)}, {make(p0, 0.5f)}));
        t.assert_true("diff ptr",     !are_lora_equal({make(p0, 1.0f)}, {make(p1, 1.0f)}));
        t.assert_true("diff size",    !are_lora_equal({make(p0, 1.0f)}, {}));
    });

    t.test("lora_get_enabled_ids", [&](testing & t) {
        auto ids = lora_get_enabled_ids({make(p0, 0.0f), make(p1, 0.5f), make(p0, -1.0f), make(p1, 2.0f)});
        t.assert_true("ids", ids == std::vector<size_t>({1, 3}));
    });
}

//
// server_pipe
//

static void test_server_pipe(testing & t) {
    auto stop_now = []() { return true; };

    t.test("drain then eof", [](testing & t) {
        server_pipe<int> p;
        t.assert_true("write 1", p.write(1));
        t.assert_true("write 2", p.write(2));
        p.close_write();

        int v = 0;
        t.assert_true("read 1", p.read(v, nullptr) && v == 1);
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("broken pipe", [](testing & t) {
        server_pipe<int> p;
        p.close_read();
        t.assert_true("write fails", !p.write(1));
    });

    t.test("max_size drops oldest", [](testing & t) {
        server_pipe<int> p;
        p.max_size = 2;
        p.write(1);
        p.write(2);
        p.write(3);
        p.close_write();

        int v = 0;
        t.assert_true("read 2", p.read(v, nullptr) && v == 2);
        t.assert_true("read 3", p.read(v, nullptr) && v == 3);
        t.assert_true("eof", !p.read(v, nullptr));
    });

    t.test("queued data wins over should_stop", [&](testing & t) {
        server_pipe<int> p;
        p.write(1);
        int v = 0;
        t.assert_true("read", p.read(v, stop_now) && v == 1);
    });

    t.test("should_stop closes the pipe", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now));
        t.assert_true("write fails", !p.write(1));
    });

    t.test("should_stop as deadline keeps the pipe open", [&](testing & t) {
        server_pipe<int> p;
        int v = 0;
        t.assert_true("read stopped", !p.read(v, stop_now, /* close_on_stop */ false));
        t.assert_true("write ok", p.write(1));
        t.assert_true("read again", p.read(v, nullptr) && v == 1);
    });

    t.test("cross thread", [](testing & t) {
        constexpr int n = 1000;
        server_pipe<int> p;
        std::thread producer([&]() {
            for (int i = 0; i < n; i++) {
                p.write(std::move(i));
            }
            p.close_write();
        });

        int v = 0;
        int count = 0;
        bool in_order = true;
        while (p.read(v, nullptr)) {
            in_order &= v == count;
            count++;
        }
        producer.join();

        t.assert_equal("count", n, count);
        t.assert_true("in order", in_order);
    });
}

//
// server_decision_escape_special_tokens
//

// "<|name|>" with U+00A6 in place of "|"
static std::string escaped(const std::string & name) {
    return "<\xC2\xA6" + name + "\xC2\xA6>";
}

static void test_decision_escape(testing & t) {
    t.test("cases", [](testing & t) {
        const std::vector<std::pair<std::string, std::string>> cases = {
            { "",                 ""                               },
            { "no tokens",        "no tokens"                      },
            { "<|im_start|>user", escaped("im_start") + "user"     },
            { "<|a_1|><|B|>",     escaped("a_1") + escaped("B")    },
            { "<|<|a|>",          "<|" + escaped("a")              },
            { "<|a|>|>",          escaped("a") + "|>"              },
            { "<||>",             "<||>"                           },
            { "<|a b|>",          "<|a b|>"                        },
            { "<|a|b|>",          "<|a|b|>"                        },
            { "<|caf\xC3\xA9|>",  "<|caf\xC3\xA9|>"                },
            { "<|abc",            "<|abc"                          },
        };
        for (const auto & [text, expected] : cases) {
            t.assert_equal(text, expected, server_decision_escape_special_tokens(text));
        }
    });

    // every string of up to 6 chars from "<|>a-" gives the text of the regex
    t.test("same as std::regex", [](testing & t) {
        static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
        const std::string chars = "<|>a-";
        size_t n_strings = 0;
        size_t n_diff    = 0;
        for (size_t len = 0; len <= 6; len++) {
            size_t n = 1;
            for (size_t k = 0; k < len; k++) {
                n *= chars.size();
            }
            for (size_t idx = 0; idx < n; idx++) {
                std::string text;
                for (size_t k = 0, v = idx; k < len; k++, v /= chars.size()) {
                    text += chars[v % chars.size()];
                }
                if (server_decision_escape_special_tokens(text) != std::regex_replace(text, re_special, "<\xC2\xA6$1\xC2\xA6>")) {
                    n_diff++;
                }
                n_strings++;
            }
        }
        t.assert_equal("strings that differ, of " + std::to_string(n_strings), (size_t) 0, n_diff);
    });

    // std::regex overflowed the stack on these
    t.test("long input", [](testing & t) {
        const std::string name(1000000, 'a');
        t.assert_true("not closed", server_decision_escape_special_tokens("<|" + name) == "<|" + name);
        t.assert_true("closed", server_decision_escape_special_tokens("<|" + name + "|>") == escaped(name));
    });
}

static void test_rerank_template(testing & t) {
    struct test_case {
        const char * name;
        std::string tmpl;
        std::string query;
        std::string doc;
        std::string expected;
    };
    const test_case cases[] = {
        {"plain inputs", "Q: {query}\nD: {document}", "query", "doc", "Q: query\nD: doc"},
        {"document placeholder in query", "Q: {query}\nD: {document}", "Explain {document}.", "doc", "Q: Explain {document}.\nD: doc"},
        {"query placeholder in query", "{query}|{document}", "{query}", "doc", "{query}|doc"},
        {"placeholders in both inputs", "{query}|{document}", "{query} {document}", "{query} {document}", "{query} {document}|{query} {document}"},
        {"repeated placeholders", "{query}|{document}|{query}|{document}", "{document}", "doc", "{document}|doc|{document}|doc"},
        {"adjacent reversed placeholders", "{document}{query}", "query", "doc", "docquery"},
        {"query only", "{query}", "{document}", "doc", "{document}"},
        {"document only", "{document}", "query", "{query}", "{query}"},
        {"no placeholders", "fixed", "{document}", "{query}", "fixed"},
        {"unknown and incomplete placeholders", "{unknown}|{query}|{document}|{query", "query", "doc", "{unknown}|query|doc|{query"},
        {"nested braces", "{{query}}|{{document}}", "query", "doc", "{query}|{doc}"},
        {"placeholder formed at boundary", "{query}ument}", "{doc", "doc", "{document}"},
        {"empty inputs", "Q: {query}\nD: {document}", "", "", "Q: \nD: "},
        {"empty template", "", "query", "doc", ""},
    };
    for (const auto & test : cases) {
        t.assert_equal(test.name, test.expected, format_rerank_template(test.tmpl, test.query, test.doc));
    }
}

//
// server_decision_context: the recursive functions stop at RECURSIVE_LIMIT
//

static void test_decision_depth(testing & t) {
    const auto prepare = [](size_t depth) {
        server_decision_context ctx;
        ctx.type = COMMON_DECISION_TYPE_H2O_LIGHTNING; // the state goes through decision_py_dump
        std::string error;
        try {
            ctx.prepare_state(json::parse(std::string(depth, '[') + std::string(depth, ']')));
        } catch (const std::invalid_argument & e) {
            error = e.what();
        }
        return error;
    };
    t.test("state at the limit", [&](testing & t) {
        const std::string error = prepare(128);
        t.assert_true("accepted: " + error, error.empty());
    });
    t.test("deep state", [&](testing & t) {
        const std::string error = prepare(1000);
        t.assert_true("rejected as nested: " + error, error.find("nested more than 128 levels deep") != std::string::npos);
    });
}

// DRY sequence breakers
//

static void test_dry_sequence_breakers(testing & t) {
    common_params params_base;

    const auto eval = [&](const json & data) {
        return server_schema::eval_llama_cmpl_schema(nullptr, params_base, {}, data);
    };

    {
        json data;
        data["dry_sequence_breakers"] = json::array({"ab", "ab", "cd", "ab"});
        const auto params = eval(data);
        const std::vector<std::string> expected = {"ab", "cd"};
        t.assert_true("duplicates removed, order kept", params.sampling.dry_sequence_breakers == expected);
    }

    {
        std::vector<std::string> many;
        for (int i = 0; i <= LLAMA_DRY_MAX_SEQ_BREAKERS; ++i) {
            many.push_back("b" + std::to_string(i));
        }
        json data;
        data["dry_sequence_breakers"] = many;

        bool thrown = false;
        try {
            eval(data);
        } catch (const std::exception &) {
            thrown = true;
        }
        t.assert_true("more than the cap is rejected", thrown);
    }
}

//
// glob_match
//

static void test_glob_match(testing & t) {
    t.test("patterns", [](testing & t) {
        t.assert_true("literal", glob_match("src/main.cpp", "src/main.cpp"));
        t.assert_true("star", glob_match("src/*.cpp", "src/main.cpp"));
        t.assert_true("star stops at /", !glob_match("*.cpp", "src/main.cpp"));
        t.assert_true("double star", glob_match("**/*.cpp", "a/b/c/main.cpp"));
        t.assert_true("double star at end", glob_match("src/**", "src/a/b"));
        t.assert_true("question mark", glob_match("ma?n.cpp", "main.cpp"));
        t.assert_true("question mark not /", !glob_match("a?b", "a/b"));
        t.assert_true("class", glob_match("[a-c]x", "bx"));
        t.assert_true("negated class", !glob_match("[!a-c]x", "bx"));
        t.assert_true("unclosed class", glob_match("[ab", "[ab"));
        t.assert_true("mismatch", !glob_match("main.cpp", "main.c"));
    });
    t.test("long input", [](testing & t) {
        const std::string path(100000, 'a');
        t.assert_true("long literal", glob_match(path, path));
        t.assert_true("long path", glob_match("**", path));
        t.assert_true("too many wildcards", !glob_match(std::string(1000, '*'), "a"));
    });
}

static void test_adaptive_draft(testing & t) {
    common_params params;
    params.n_parallel = 1;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
    params.speculative.draft.probabilistic = true;
    params.speculative.draft.n_max = 4;
    server_adaptive_draft_costs data;
    data.history_tokens = 512;
    data.temperature = params.sampling.temp;
    data.top_p = params.sampling.top_p;
    data.top_k = params.sampling.top_k;
    data.min_p = params.sampling.min_p;
    for (int d = 1; d <= 4; ++d) {
        data.rows.push_back({d, 3, 1000.0, 0.0});
    }
    data.rows.push_back({0, 3, 100000.0, 0.0});
    t.test("robust startup cost rejects isolated outliers", [](testing & t) {
        for (int draft = 0; draft <= 7; ++draft) {
            const auto cost = server_adaptive_draft::summarize_cost(draft, {99, 100, 101, 100, 99, 101, 9000});
            t.assert_equal("robust center", 100.0, cost.mean_us);
            t.assert_equal("all repetitions recorded", 7, cost.samples);
            t.assert_true("outlier does not inflate timing uncertainty", cost.sd_us < 2.0);
        }
        int failed = 0;
        for (const auto & values : std::vector<std::vector<double>>{{}, {0}, {-1}, {INFINITY}, {NAN}}) {
            try { server_adaptive_draft::summarize_cost(1, values); } catch (const std::invalid_argument &) { ++failed; }
        }
        t.assert_equal("invalid timing samples rejected", 5, failed);
    });
    t.test("promoted acceptance regression rechecks measured alternatives", [&](testing & t) {
        for (int trial = 0; trial < 28; ++trial) {
            const int scenario = trial / 7;
            const int promoted = trial % 7 + 1;
            const int alternative = promoted == 3 ? 1 : 3;
            auto options = params;
            options.speculative.draft.n_max = 7;
            auto costs = data;
            costs.rows.clear();
            for (int d = 0; d <= 7; ++d) {
                costs.rows.push_back({d, 7, d == promoted || (promoted == 3 && d == 1) ? 100.0 : d == 3 ? 1000.0 : 1e6, 0.0});
            }
            server_adaptive_draft runtime(costs, options);
            int64_t now = 100;
            bool reached = false;
            for (int i = 0; i < 128 && !reached; ++i) {
                const int d = runtime.begin(7, now);
                const int accepted = d == promoted ? d : std::min(d, 1);
                now += d == promoted && promoted < 3 ? 375 * (d + 1) : 1000;
                if (runtime.observe(d, accepted, accepted + 1, false)) {
                    runtime.drain(now);
                    const auto state = runtime.status();
                    reached = state.at("incumbent") == promoted && (promoted != 3 ||
                            (state.at("probe_draft") == 1 && !state.at("probe").get<bool>()));
                }
            }
            t.assert_true("candidate promoted D" + std::to_string(promoted), reached);
            if (!reached) { runtime.abandon(now + 1); continue; }
            int steady = 0;
            bool rechecked = false;
            for (int i = 0; i < 48 && !rechecked; ++i) {
                const auto before = runtime.status();
                const int d = runtime.begin(7, now, scenario == 3 ? 8 : 512);
                const int accepted = d == promoted ? (scenario == 2 && steady >= 8 ? d : 0) : std::min(d, 1);
                now += d == promoted && promoted < 3 ? 375 * (d + 1) : 1000;
                if (d == promoted && !before.at("probe").get<bool>()) { ++steady; }
                if (runtime.observe(d, accepted, accepted + 1, false)) {
                    runtime.drain(now);
                    if (steady == 8) { t.assert_equal("one low window is insufficient", 0, runtime.status().at("acceptance_refreshes").get<int>()); }
                    rechecked = runtime.status().at("recovery_probes").get<int>() > 0;
                    if (scenario >= 2 && steady >= 24) { break; }
                }
            }
            if (scenario >= 2) {
                t.assert_true("single dip or short horizon avoids recovery probe", !rechecked);
                if (scenario == 2) { t.assert_equal("recovered acceptance keeps historical evidence", 0, runtime.status().at("acceptance_refreshes").get<int>()); }
                runtime.abandon(now + 1);
                continue;
            }
            t.assert_true("two low windows trigger recovery D" + std::to_string(promoted), rechecked);
            if (rechecked) {
                t.assert_equal("recovery verifies a measured candidate", alternative, runtime.status().at("draft").get<int>());
                for (int i = 0; i < 8; ++i) {
                    const int d = runtime.begin(7, now);
                    now += scenario == 1 ? 100000 : 1000;
                    const int accepted = scenario == 1 ? 0 : 1;
                    if (runtime.observe(d, accepted, accepted + 1, false)) { runtime.drain(now); }
                }
                t.assert_equal("recovery uses fresh measured speed", scenario == 1 ? promoted : alternative, runtime.status().at("incumbent").get<int>());
                t.assert_equal("recovery promotion is recorded", scenario == 1 ? 0 : 1, runtime.status().at("recovery_switches").get<int>());
                if (scenario == 1) {
                    t.assert_true("failed recovery starts a new cooldown", runtime.status().at("cooldown_tokens").at(alternative).get<int>() >= 128);
                    t.assert_true("failed recovery restores normal mode", !runtime.status().at("probe").get<bool>());
                }
            }
            runtime.abandon(now + 1);
        }
    });
    t.test("recovery ranks new posterior above old speed", [&](testing & t) {
        auto options = params;
        options.speculative.draft.n_max = 7;
        options.sampling.seed = 1;
        auto costs = data;
        costs.rows = {
            {0, 7, 23747.000, 620.000},
            {1, 7, 45736.000, 2466.000},
            {2, 7, 47449.000, 2034.000},
            {3, 7, 51435.000, 1451.000},
            {4, 7, 57716.000, 2391.000},
            {5, 7, 68368.000, 1916.000},
            {6, 7, 70000.000, 494.000},
            {7, 7, 71695.000, 2839.000},
        };
        struct frame { int draft; int64_t elapsed; int64_t boundary; std::vector<int> accepted; };
        const std::vector<frame> frames = {
            {3, 967953, 0, {3, 2, 1, 2, 3, 3, 1, 1, 0, 2, 2, 3, 0, 0, 1, 3}},
            {4, 829987, 2693, {2, 4, 1, 4, 1, 4, 0, 0}},
            {3, 939545, 3283, {1, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
            {3, 499137, 2372, {0, 0, 1, 1, 0, 1, 1, 0}},
        };
        server_adaptive_draft runtime(costs, options);
        int64_t now = 100;
        for (const auto & frame : frames) {
            now += frame.boundary;
            for (size_t i = 0; i < frame.accepted.size(); ++i) {
                const int remaining = 511 - runtime.status().at("committed_tokens").get<int>();
                const int d = runtime.begin(7, now, remaining);
                t.assert_equal("replay follows measured blocks", frame.draft, d);
                now += frame.elapsed / frame.accepted.size();
                if (i + 1 == frame.accepted.size()) { now += frame.elapsed % frame.accepted.size(); }
                if (runtime.observe(d, frame.accepted[i], frame.accepted[i] + 1, false)) { runtime.drain(now); }
            }
        }
        const auto state = runtime.status();
        t.assert_equal("new regime is detected", 1, state.at("acceptance_refreshes").get<int>());
        t.assert_equal("unmeasured better candidate participates", 0, state.at("draft").get<int>());
        t.assert_equal("recovery is verified before promotion", std::string("acceptance_recovery_probe"), state.at("reason").get<std::string>());
        t.assert_true("old measured speed has no fresh posterior support", state.at("predictions").at(4).at("gain").get<double>() < 0.0);
        t.assert_true("chosen plan has positive predicted savings", state.at("recovery_expected_saving_us").get<double>() > 0.0);
        runtime.abandon(now + 1);
    });
    t.test("actual forward length", [](testing & t) {
        common_speculative_draft_params dp;
        dp.n_max = 3;
        t.assert_equal("default keeps fixed block", 7, common_speculative_actual_draft_length(dp, 7));
        dp.actual_draft_length = true;
        for (int d : {7, 1, 3, 6, 2, 5, 4}) {
            dp.n_max = d;
            t.assert_equal("forward uses requested depth", d, common_speculative_actual_draft_length(dp, 7));
        }
        int failed = 0;
        for (int d : {-1, 0, 8}) {
            dp.n_max = d;
            try { common_speculative_actual_draft_length(dp, 7); } catch (const std::invalid_argument &) { ++failed; }
        }
        t.assert_equal("invalid lengths rejected", 3, failed);
    });
    t.test("equal costs promote a longer block without credit", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        int64_t now = 100;
        int first_probe = 0;
        for (int round = 1; round <= 64 && runtime.status().at("incumbent") != 4; ++round) {
            const int d = runtime.begin(4, now);
            if (runtime.status().at("probe").get<bool>() && first_probe == 0) { first_probe = round; }
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("first normal window triggers a probe", 17, first_probe);
        t.assert_equal("25 percent improvement becomes incumbent", 4, runtime.status().at("incumbent").get<int>());
        t.assert_true("switch uses recent direct evidence", runtime.status().at("probe_rounds").get<int>() >= 8);
        t.assert_true("switch ends temporary probing", !runtime.status().at("probe").get<bool>());
        runtime.abandon(now + 1);
    });
    t.test("normal cost ratios permit a shorter block", [&](testing & t) {
        auto cheaper = data;
        for (auto & row : cheaper.rows) { if (row.draft > 0) { row.mean_us = 750.0 + 250.0 * row.draft; } }
        server_adaptive_draft runtime(cheaper, params);
        int64_t now = 100;
        for (int round = 0; round < 128 && runtime.status().at("incumbent") != 1; ++round) {
            const int d = runtime.begin(4, now);
            now += 750 + 250 * d;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("measured shorter block becomes incumbent", 1, runtime.status().at("incumbent").get<int>());
        runtime.abandon(now + 1);
    });
    t.test("bad probe stops and cools down", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        int64_t now = 100;
        bool stopped = false;
        for (int round = 0; round < 64 && !stopped; ++round) {
            const int d = runtime.begin(4, now);
            const int accepted = d == 4 ? 0 : d;
            now += 1000;
            if (runtime.observe(d, accepted, accepted + 1, false)) {
                const bool was_probe = runtime.status().at("probe").get<bool>();
                runtime.drain(now);
                stopped = was_probe && !runtime.status().at("probe").get<bool>();
            }
        }
        t.assert_true("unprofitable probe was stopped", stopped);
        t.assert_equal("incumbent restored", 3, runtime.status().at("incumbent").get<int>());
        t.assert_equal("failed candidate cooldown starts", 128, runtime.status().at("cooldown_tokens").at(4).get<int>());
        for (int round = 0; round < 16; ++round) {
            const int d = runtime.begin(4, now);
            t.assert_equal("failed candidate is not retried during cooldown", 3, d);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("cooldown ages with committed tokens", 64, runtime.status().at("cooldown_tokens").at(4).get<int>());
        runtime.abandon(now + 1);
    });
    t.test("uncertain probes are bounded and back off", [&](testing & t) {
        auto noisy = data;
        for (auto & row : noisy.rows) {
            row.sd_us = 350.0;
            if (row.draft == 1 || row.draft == 2) { row.mean_us = 1000000.0; }
        }
        server_adaptive_draft runtime(noisy, params);
        int64_t now = 100;
        int attempts = 0;
        for (int round = 0; round < 256 && attempts < 2; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) {
                const bool was_probe = runtime.status().at("probe").get<bool>();
                runtime.drain(now);
                if (was_probe && !runtime.status().at("probe").get<bool>()) {
                    ++attempts;
                    t.assert_equal("inconclusive episode is capped", 16, runtime.status().at("probe_rounds").get<int>());
                    t.assert_equal("episode ended at round limit", std::string("round_limit"), runtime.status().at("probe_outcome").get<std::string>());
                    t.assert_equal("repeated failures lengthen cooldown", 128 * attempts, runtime.status().at("cooldown_tokens").at(4).get<int>());
                }
            }
        }
        t.assert_equal("candidate can be reconsidered after cooldown", 2, attempts);
        t.assert_equal("uncertainty prevents premature promotion", 3, runtime.status().at("incumbent").get<int>());
        runtime.abandon(now + 1);
    });
    t.test("credible probe can promote at the round limit", [&](testing & t) {
        auto noisy = data;
        for (auto & row : noisy.rows) { row.sd_us = 200.0; }
        server_adaptive_draft runtime(noisy, params);
        int64_t now = 100;
        for (int round = 0; round < 32; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
            if (round == 23) { t.assert_true("eight rounds still need evidence", runtime.status().at("probe").get<bool>()); }
        }
        t.assert_equal("measured gain can promote at sixteen", 4, runtime.status().at("incumbent").get<int>());
        t.assert_equal("cap decision uses a bounded evidence threshold", std::string("bounded_probe_gain"), runtime.status().at("reason").get<std::string>());
        t.assert_equal("episode does not exceed its cap", 16, runtime.status().at("probe_rounds").get<int>());
        runtime.abandon(now + 1);
    });
    t.test("phase change can reverse a prior switch", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        int64_t now = 100;
        bool reached_four = false;
        bool returned = false;
        for (int round = 0; round < 2000 && !returned; ++round) {
            const int d = runtime.begin(4, now);
            const int accepted = reached_four && d == 4 ? 0 : d;
            now += 1000;
            if (runtime.observe(d, accepted, accepted + 1, false)) {
                runtime.drain(now);
                const int incumbent = runtime.status().at("incumbent").get<int>();
                if (incumbent == 4) { reached_four = true; }
                else if (reached_four) { returned = true; }
            }
        }
        t.assert_true("easy phase selected longer block", reached_four);
        t.assert_true("changed acceptance allowed a shorter incumbent", returned);
        runtime.abandon(now + 1);
    });
    t.test("boundary overhead does not inflate short probe cost", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        int64_t now = 100;
        for (int round = 0; round < 16; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_true("probe scheduled", runtime.status().at("probe").get<bool>());
        now += 100000;
        for (int round = 0; round < 8; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("probe cost excludes boundary work", 1000.0, runtime.status().at("round_us").get<double>());
        t.assert_equal("boundary work remains accounted", 100000.0, runtime.status().at("boundary_us").get<double>());
        t.assert_equal("runtime includes all elapsed work", 124000.0, runtime.status().at("runtime_us").get<double>());
        t.assert_equal("probe uses normal-window amortization", 6250.0, runtime.status().at("round_overhead_us").get<double>());
        now += 40000;
        for (int round = 0; round < 16; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("boundary estimate follows new measurements", 85000.0, runtime.status().at("boundary_mean_us").get<double>());
        t.assert_equal("same amortization after promotion", 5312.5, runtime.status().at("round_overhead_us").get<double>());
        t.assert_equal("new boundary work is accounted once", 180000.0, runtime.status().at("runtime_us").get<double>());
        t.assert_equal("inference timing stays separate", 1000.0, runtime.status().at("round_us").get<double>());
        runtime.abandon(now + 1);
    });
    t.test("ordinary promotion respects recent measured throughput", [&](testing & t) {
        auto costs = data;
        for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 260.0 : row.draft == 3 ? 1000.0 : 1000000.0; }
        server_adaptive_draft runtime(costs, params);
        int64_t now = 100;
        for (int i = 0; i < 24; ++i) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 260 : 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("slower ordinary probe cannot become incumbent", 3, runtime.status().at("incumbent").get<int>());
        t.assert_equal("guard explains rejected posterior promotion", std::string("ordinary_rate_guard"), runtime.status().at("reason").get<std::string>());
        runtime.abandon(now + 1);
    });
    t.test("fresh ordinary evidence can end cooldown without another probe", [&](testing & t) {
        auto costs = data;
        for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 500.0 : row.draft == 3 ? 1000.0 : 1000000.0; }
        server_adaptive_draft runtime(costs, params);
        int64_t now = 100;
        bool failed_probe = false;
        uint64_t failed_at = 0;
        for (int i = 0; i < 160 && runtime.status().at("incumbent") != 0; ++i) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 1500 : failed_probe ? 3000 : 1000;
            if (runtime.observe(d, 0, 1, false)) {
                const bool was_probe = runtime.status().at("probe").get<bool>();
                runtime.drain(now);
                if (was_probe && !runtime.status().at("probe").get<bool>() && runtime.status().at("incumbent") == 3) {
                    failed_probe = true;
                    failed_at = runtime.status().at("committed_tokens").get<uint64_t>();
                }
            }
        }
        t.assert_true("initial slower D0 probe was rejected", failed_probe);
        t.assert_equal("slower incumbent now falls back", 0, runtime.status().at("incumbent").get<int>());
        t.assert_equal("fresh evidence released ordinary cooldown", 1, runtime.status().at("ordinary_cooldown_releases").get<int>());
        t.assert_true("switch happened before cooldown expiry", runtime.status().at("committed_tokens").get<uint64_t>() - failed_at < 128);
        t.assert_equal("no repeat probe was needed", 1, runtime.status().at("probe_episodes").get<int>());
        runtime.abandon(now + 1);
    });
    t.test("recent throughput rechecks stale ordinary cost only when it can repay", [&](testing & t) {
        for (int64_t recheck_cost : {335, 450, 0}) {
            const int remaining = recheck_cost == 0 ? 16 : 512;
            auto costs = data;
            for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 200.0 : row.draft == 3 ? 1000.0 : 1000000.0; }
            server_adaptive_draft runtime(costs, params);
            int64_t now = 100;
            auto run = [&](int depth, int rounds, int64_t round_us, int64_t transition = 0) {
                for (int i = 0; i < rounds; ++i) {
                    const int d = runtime.begin(4, now, remaining);
                    t.assert_equal("expected depth", depth, d);
                    if (i == 0 && transition > 0) {
                        runtime.account_transition(transition);
                        now += transition;
                    }
                    now += round_us;
                    if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
                }
            };
            run(3, 16, 1000);
            run(0, 1, 385, 350);
            run(0, 8, 335);
            t.assert_equal("D0 loses to the initial D3 rate", 3, runtime.status().at("incumbent").get<int>());
            run(3, 16, 1000, 600);
            run(3, 32, 1000);
            run(3, 16, 1550);
            t.assert_true("one slow window cannot recheck D0", !runtime.status().at("probe").get<bool>());
            run(3, 16, 1550);
            const auto state = runtime.status();
            t.assert_true("long history still disfavors D0", state.at("predictions").at(0).at("probability_gain").get<double>() < 0.60);
            if (recheck_cost > 0) {
                t.assert_equal("sustained slowdown earns a fresh measurement", std::string("ordinary_recheck"), state.at("reason").get<std::string>());
                run(0, 1, 385, 350);
                run(0, 8, recheck_cost);
                const bool confirmed = recheck_cost == 335;
                t.assert_equal("fresh D0 speed determines fallback", confirmed ? 0 : 3, runtime.status().at("incumbent").get<int>());
                const std::string reason = confirmed ? "ordinary_recheck_gain" : "ordinary_recheck_no_gain";
                t.assert_equal("decision uses the new measurement", reason, runtime.status().at("reason").get<std::string>());
                t.assert_true("recheck ends after eight stable rounds", !runtime.status().at("probe").get<bool>());
            } else {
                t.assert_true("too few tokens remain to pay for recheck", !state.at("probe").get<bool>());
                t.assert_equal("request stays at D3", 3, state.at("incumbent").get<int>());
            }
            runtime.abandon(now + 1);
        }
    });
    t.test("measured D0 gain takes priority over a shorter positive probe", [&](testing & t) {
        auto options = params;
        options.sampling.seed = 0;
        options.speculative.draft.n_max = 2;
        auto costs = data;
        costs.rows = {{0, 3, 570.0, 0.0}, {1, 3, 1375.0, 0.0}, {2, 3, 1000.0, 0.0}};
        server_adaptive_draft runtime(costs, options);
        int64_t now = 100;
        auto run = [&](int depth, int rounds, int64_t round_us) {
            for (int i = 0; i < rounds; ++i) {
                const int d = runtime.begin(2, now);
                t.assert_equal("expected depth", depth, d);
                now += round_us;
                if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
            }
        };
        run(2, 16, 1000);
        run(0, 8, 1600);
        run(2, 32, 1000);
        run(2, 32, 2200);
        const auto state = runtime.status();
        const auto candidate = state.at("predictions").at(1);
        t.assert_true("D1 qualifies for a regular probe",
                candidate.at("probability_gain").get<double>() >= 0.60 && candidate.at("gain").get<double>() > 0.03);
        t.assert_equal("measured D0 wins over D1 exploration", 0, state.at("incumbent").get<int>());
        t.assert_equal("fallback uses the measured rate", std::string("ordinary_recent_gain"), state.at("reason").get<std::string>());
        t.assert_equal("fallback does not start another probe", 1, state.at("probe_episodes").get<int>());
        runtime.abandon(now + 1);
    });
    t.test("unmeasured D0 can precede a longer probe after incumbent slowdown", [&](testing & t) {
        auto options = params;
        options.sampling.seed = 0;
        auto costs = data;
        costs.rows = {{0, 3, 275.0, 0.0}, {1, 3, 1000000.0, 0.0}, {2, 3, 1000000.0, 0.0},
                      {3, 3, 1000.0, 0.0}, {4, 3, 1200.0, 0.0}};
        for (int64_t ordinary_us : {220, 1200}) {
            server_adaptive_draft runtime(costs, options);
            int64_t now = 100;
            auto run = [&](int depth, int rounds, int accepted, int64_t round_us) {
                for (int i = 0; i < rounds; ++i) {
                    const int d = runtime.begin(4, now);
                    t.assert_equal("expected depth", depth, d);
                    now += round_us;
                    if (runtime.observe(d, accepted, accepted + 1, false)) { runtime.drain(now); }
                }
            };
            run(3, 16, 3, 1000);
            t.assert_equal("first window keeps incumbent", std::string("hold"), runtime.status().at("action").get<std::string>());
            run(3, 16, 1, 1750);
            const auto state = runtime.status();
            const auto candidate = state.at("predictions").at(4);
            t.assert_true("D4 qualifies for a regular probe",
                    candidate.at("probability_gain").get<double>() >= 0.60 && candidate.at("gain").get<double>() > 0.03);
            t.assert_equal("D0 is measured before D4", 0, state.at("probe_draft").get<int>());
            t.assert_equal("strong prediction only schedules a probe", 3, state.at("incumbent").get<int>());
            t.assert_equal("decision reports the probe priority", std::string("ordinary_probe_priority"), state.at("reason").get<std::string>());
            run(0, 8, 0, ordinary_us);
            t.assert_equal("D0 must confirm its gain in a real window", ordinary_us == 220 ? 0 : 3,
                    runtime.status().at("incumbent").get<int>());
            runtime.abandon(now + 1);
        }
    });
    t.test("terminal limit and request reset", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        t.assert_equal("tail is actually shortened", 1, runtime.begin(1, 100));
        t.assert_true("terminal round drains", runtime.observe(1, 1, 1, true));
        runtime.drain(1100, true);
        runtime.abandon(1200);
        t.assert_true("terminal cost is excluded", !runtime.status().at("cost_usable").get<bool>());
        server_adaptive_draft next(data, params);
        t.assert_equal("new request starts at D3", 3, next.begin(4, 2000));
        t.assert_equal("new request has no probe history", 0, next.status().at("probe_episodes").get<int>());
        next.abandon(2100);
    });
    t.test("cancel first probe round accounts actual time", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        int64_t now = 100;
        for (int round = 0; round < 16; ++round) {
            const int d = runtime.begin(4, now);
            now += 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
        }
        t.assert_true("probe scheduled", runtime.status().at("probe").get<bool>());
        runtime.begin(4, now);
        runtime.abandon(now + 500);
        t.assert_equal("cancel invents no acceptance round", 16, runtime.status().at("rounds").get<int>());
        t.assert_equal("cancelled probe time accounted", 500.0, runtime.status().at("probe_us").get<double>());
        t.assert_true("cancel is not a normal cost observation", !runtime.status().at("cost_usable").get<bool>());
        t.assert_true("cancel clears pending probe", !runtime.status().at("probe").get<bool>());
    });
    t.test("actual block mismatch rejected", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        runtime.begin(7, 100);
        bool failed = false;
        try { runtime.observe(7, 0, 1, false); } catch (const std::logic_error &) { failed = true; }
        t.assert_true("fixed D7 block cannot pretend to be D3", failed);
        runtime.abandon(1000);
    });
    t.test("ordinary fallback, resume and context shift", [&](testing & t) {
        auto costs = data;
        for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 500.0 : 2000.0; }
        server_adaptive_draft runtime(costs, params);
        int64_t now = 100;
        for (int round = 0; round < 128 && runtime.status().at("incumbent") != 0; ++round) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 500 : 2000;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("ordinary becomes incumbent", 0, runtime.status().at("incumbent").get<int>());
        t.assert_equal("D0 has direct probe evidence", 8, runtime.status().at("probe_rounds").get<int>());
        bool resumed = false;
        for (int round = 0; round < 512 && !resumed; ++round) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 10000 : 1000;
            if (runtime.observe(d, d, d + 1, false)) { runtime.drain(now); }
            resumed = runtime.status().at("incumbent").get<int>() > 0;
        }
        t.assert_true("changed costs permit DSpark reentry", resumed);
        runtime.stay_ordinary();
        for (int round = 0; round < 32; ++round) {
            t.assert_equal("shift locks D0 until request end", 0, runtime.begin(4, now));
            now += 10000;
            if (runtime.observe(0, 0, 1, false)) { runtime.drain(now); }
        }
        runtime.abandon(now + 1);
    });
    t.test("ordinary aging and isolated slowdown do not reopen speculation", [&](testing & t) {
        auto costs = data;
        for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 500.0 : 2000.0; }
        server_adaptive_draft runtime(costs, params);
        int64_t now = 100;
        for (int round = 0; round < 128 && runtime.status().at("incumbent") != 0; ++round) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 500 : 2000;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("fixture falls back", 0, runtime.status().at("incumbent").get<int>());
        const auto episodes = runtime.status().at("probe_episodes").get<uint64_t>();
        for (int round = 0; round < 544; ++round) {
            const int d = runtime.begin(4, now);
            t.assert_equal("aging alone keeps ordinary decoding", 0, d);
            now += round >= 512 && round < 528 ? 1000 : 500;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("neither stale evidence nor one slow window restarts DSpark", episodes, runtime.status().at("probe_episodes").get<uint64_t>());
        t.assert_true("suppressed reentry windows are visible", runtime.status().at("reentry_blocked_windows").get<int>() > 0);
        runtime.abandon(now + 1);
    });
    t.test("failed reentry needs another ordinary cost change", [&](testing & t) {
        auto costs = data;
        for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 500.0 : 2000.0; }
        server_adaptive_draft runtime(costs, params);
        int64_t now = 100;
        for (int round = 0; round < 128 && runtime.status().at("incumbent") != 0; ++round) {
            const int d = runtime.begin(4, now);
            now += d == 0 ? 500 : 2000;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        bool failed_reentry = false;
        for (int round = 0; round < 128 && !failed_reentry; ++round) {
            const int d = runtime.begin(4, now);
            const bool reentry = d > 0;
            now += reentry ? 100000 : 10000;
            if (runtime.observe(d, 0, 1, false)) {
                runtime.drain(now);
                failed_reentry = reentry && !runtime.status().at("probe").get<bool>();
            }
        }
        t.assert_true("cost change tried and rejected an expensive restart", failed_reentry);
        const auto episodes = runtime.status().at("probe_episodes").get<uint64_t>();
        for (int round = 0; round < 160; ++round) {
            const int d = runtime.begin(4, now);
            t.assert_equal("unchanged ordinary cost cannot retry other depths", 0, d);
            now += 10000;
            if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
        }
        t.assert_equal("failed reentry resets the evidence reference", episodes, runtime.status().at("probe_episodes").get<uint64_t>());
        runtime.abandon(now + 1);
    });
    t.test("ordinary entry warmup is charged without poisoning steady cost", [&](testing & t) {
        for (bool profitable : {true, false}) {
            auto costs = data;
            for (auto & row : costs.rows) { row.mean_us = row.draft == 0 ? 500.0 : 2000.0; }
            server_adaptive_draft runtime(costs, params);
            int64_t now = 100;
            for (int round = 0; round < 128 && runtime.status().at("draft") != 0; ++round) {
                const int d = runtime.begin(4, now);
                now += 2000;
                if (runtime.observe(d, 0, 1, false)) { runtime.drain(now); }
            }
            t.assert_true("ordinary probe scheduled", runtime.status().at("probe").get<bool>());
            const double prior_probe_us = runtime.status().at("probe_us").get<double>();
            t.assert_equal("probe executes D0", 0, runtime.begin(4, now));
            t.assert_true("real D0 entry changes active hooks", runtime.needs_mode_transition(true));
            runtime.account_transition(3000);
            now += 3000 + 20000;
            t.assert_true("first forward drains separately", runtime.observe(0, 0, 1, false));
            runtime.drain(now);
            t.assert_true("warmup marked", runtime.status().at("measured_warmup").get<bool>());
            t.assert_true("cold cost is not learned", !runtime.status().at("cost_usable").get<bool>());
            t.assert_equal("warmup and transition are charged", prior_probe_us + 23000.0, runtime.status().at("probe_us").get<double>());
            t.assert_equal("warmup is not steady probe evidence", 0, runtime.status().at("probe_rounds").get<int>());
            for (int round = 0; round < 8; ++round) {
                t.assert_equal("steady probe stays D0", 0, runtime.begin(4, now));
                t.assert_true("no repeated mode transition", !runtime.needs_mode_transition(false));
                now += profitable ? 500 : 20000;
                const bool drain = runtime.observe(0, 0, 1, false);
                t.assert_equal("measure eight stable rounds", round == 7, drain);
                if (drain) { runtime.drain(now); }
            }
            t.assert_equal("only stable duration trains cost", profitable ? 500.0 : 20000.0,
                    runtime.status().at("round_us").get<double>());
            t.assert_equal("steady window is usable", true, runtime.status().at("cost_usable").get<bool>());
            t.assert_equal("cost determines fallback", profitable ? 0 : 3, runtime.status().at("incumbent").get<int>());
            t.assert_equal("all probe time remains charged", prior_probe_us + 23000.0 + 8 * (profitable ? 500 : 20000),
                    runtime.status().at("probe_us").get<double>());
            runtime.abandon(now + 1);
        }
    });
    t.test("ordinary tail is timed without false acceptance", [&](testing & t) {
        server_adaptive_draft runtime(data, params);
        t.assert_equal("zero capacity opens an ordinary round", 0, runtime.begin(0, 100));
        t.assert_true("D0 is a pending round, not an empty sentinel", runtime.ordinary_pending());
        t.assert_true("tail does not save a resume checkpoint", !runtime.needs_mode_transition(true));
        t.assert_true("paused tail does not replay history", !runtime.needs_mode_transition(false));
        t.assert_true("terminal D0 drains", runtime.observe(0, 0, 1, true));
        runtime.drain(1100, true);
        t.assert_equal("one token accounted", 1, runtime.status().at("committed_tokens").get<int>());
        t.assert_equal("ordinary tail duration accounted", 1000.0, runtime.status().at("runtime_us").get<double>());
        t.assert_true("terminal tail is not normal cost evidence", !runtime.status().at("cost_usable").get<bool>());
        server_adaptive_draft cancel(data, params);
        cancel.begin(0, 100);
        cancel.abandon(200);
        t.assert_equal("cancelled D0 invents no committed token", 0, cancel.status().at("committed_tokens").get<int>());
        t.assert_equal("cancelled D0 is charged", 100.0, cancel.status().at("runtime_us").get<double>());
    });
    t.test("incompatible inputs", [&](testing & t) {
        int failed = 0;
        auto reject = [&](const server_adaptive_draft_costs & costs, const common_params & options) {
            try { server_adaptive_draft runtime(costs, options); } catch (const std::exception &) { ++failed; }
        };
        auto bad = data;
        bad.rows[2].draft = 8;
        reject(bad, params);
        bad = data;
        bad.rows[2].draft = 2;
        reject(bad, params);
        bad = data;
        bad.rows.pop_back();
        reject(bad, params);
        bad = data;
        bad.temperature += 0.1f;
        reject(bad, params);
        auto options = params;
        options.n_parallel = 2;
        reject(data, options);
        options = params;
        options.sampling.temp = 0.0f;
        reject(data, options);
        t.assert_equal("all incompatible modes rejected", 6, failed);
    });
}

static void test_fixed_d3_ffn_defaults(testing & t) {
    common_params p;
    p.n_parallel = 1;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
    p.speculative.draft.probabilistic = true;
    t.assert_true("fixed D3", server_use_fixed_d3_ffn_defaults(p));
    p.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
    t.assert_true("explicit --spec-type draft-dspark", server_use_fixed_d3_ffn_defaults(p));
    p.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
    t.assert_true("combined draft methods", !server_use_fixed_d3_ffn_defaults(p));
    p.speculative.types.pop_back();

    for (int depth : {0, 1, 2, 4, 7}) {
        p.speculative.draft.n_max = depth;
        t.assert_true("other fixed depth", !server_use_fixed_d3_ffn_defaults(p));
    }
    p.speculative.draft.n_max = 3;
    p.speculative.adaptive = true;
    t.assert_true("adaptive D3", !server_use_fixed_d3_ffn_defaults(p));
    p.speculative.adaptive = false;
    p.cache_type_v_swa = GGML_TYPE_Q8_0;
    t.assert_true("mixed SWA cache", !server_use_fixed_d3_ffn_defaults(p));
    p.cache_type_v_swa = GGML_TYPE_COUNT;
    p.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
    t.assert_true("quantized draft KV", !server_use_fixed_d3_ffn_defaults(p));
    p.speculative.draft.cache_type_k = GGML_TYPE_F16;
    p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    t.assert_true("other draft method", !server_use_fixed_d3_ffn_defaults(p));
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("server_tokens", test_server_tokens);
    t.test("utf8",          test_utf8);
    t.test("sse",           test_sse);
    t.test("json_helpers",  test_json_helpers);
    t.test("lora_helpers",  test_lora_helpers);
    t.test("server_pipe",   test_server_pipe);
    t.test("decision_escape", test_decision_escape);
    t.test("rerank_template", test_rerank_template);
    t.test("decision_depth",  test_decision_depth);
    t.test("glob_match",      test_glob_match);
    t.test("dry_sequence_breakers", test_dry_sequence_breakers);
    t.test("adaptive_draft", test_adaptive_draft);
    t.test("fixed_d3_ffn_defaults", test_fixed_d3_ffn_defaults);

    return t.summary();
}
