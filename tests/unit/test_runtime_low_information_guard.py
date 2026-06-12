import unittest

from lafla_ai_core.runtime.output_guard import sanitize_completion


LOW_INFORMATION_CASES = (
    ("dash_run", "------------------------------------------------"),
    ("spaced_dash_run", "- - - - - - - - - - - -"),
    ("dash_blocks", "-------- -------- --------"),
    ("dash_lines", "----\n----\n----\n----"),
    ("dot_tokens", ". . . . . . . . . . . ."),
    ("dot_run", "................................"),
    ("bang_tokens", "! ! ! ! ! ! ! ! ! !"),
    ("bang_run", "!!!!!!!!!!!!!!!!!!!!!!!!"),
    ("question_tokens", "? ? ? ? ? ? ? ? ? ?"),
    ("question_run", "????????????????????????"),
    ("equals_tokens", "= = = = = = = = = ="),
    ("equals_run", "========================"),
    ("underscore_tokens", "_ _ _ _ _ _ _ _ _ _"),
    ("underscore_run", "________________________"),
    ("slash_tokens", "/ / / / / / / / / /"),
    ("slash_run", "////////////////////////"),
    ("backslash_tokens", "\\ \\ \\ \\ \\ \\ \\ \\ \\ \\"),
    ("backslash_run", "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"),
    ("pipe_tokens", "| | | | | | | | | |"),
    ("pipe_run", "||||||||||||||||||||||||"),
    ("star_tokens", "* * * * * * * * * *"),
    ("star_run", "************************"),
    ("plus_tokens", "+ + + + + + + + + +"),
    ("plus_run", "++++++++++++++++++++++++"),
    ("colon_tokens", ": : : : : : : : : :"),
    ("colon_run", "::::::::::::::::::::::::"),
    ("semicolon_tokens", "; ; ; ; ; ; ; ; ; ;"),
    ("semicolon_run", ";;;;;;;;;;;;;;;;;;;;;;;;"),
    ("comma_tokens", ", , , , , , , , , ,"),
    ("comma_run", ",,,,,,,,,,,,,,,,,,,,,,,,"),
    ("tilde_tokens", "~ ~ ~ ~ ~ ~ ~ ~ ~ ~"),
    ("tilde_run", "~~~~~~~~~~~~~~~~~~~~~~~~"),
    ("backtick_tokens", "` ` ` ` ` ` ` ` ` `"),
    ("backtick_run", "````````````````````````"),
    ("quote_tokens", "' ' ' ' ' ' ' ' ' '"),
    ("quote_run", "''''''''''''''''''''''''"),
    ("paren_loop", "()()()()()()()()()()()()"),
    ("bracket_loop", "[][][][][][][][][][][][]"),
    ("brace_loop", "{}{}{}{}{}{}{}{}{}{}{}{}"),
    ("angle_loop", "<><><><><><><><><><><><>"),
    ("double_dash_tokens", "-- -- -- -- -- -- -- --"),
    ("word_ha_loop", "ha ha ha ha ha ha ha ha"),
    ("word_tamam_loop", "tamam tamam tamam tamam tamam tamam tamam"),
    ("word_evet_loop", "evet evet evet evet evet evet evet"),
    ("word_hayir_loop", "hayir hayir hayir hayir hayir hayir hayir"),
    ("word_lafla_loop", "Lafla Lafla Lafla Lafla Lafla Lafla"),
    ("word_cevap_loop", "cevap cevap cevap cevap cevap cevap"),
    ("number_two_loop", "2 2 2 2 2 2 2 2 2"),
    ("number_four_loop", "4 4 4 4 4 4 4 4 4"),
    ("letter_a_run", "aaaaaaaaaaaaaaaaaaaaaaaa"),
    ("letter_k_run", "kkkkkkkkkkkkkkkkkkkkkkkk"),
    ("number_one_run", "111111111111111111111111"),
    ("number_zero_run", "000000000000000000000000"),
    ("abc_word_loop", "abc abc abc abc abc abc abc"),
    ("two_word_alternation", "la fla la fla la fla la fla la fla"),
    ("math_fragment_loop", "2 + 2 2 + 2 2 + 2 2 + 2"),
    ("english_the_loop", "the the the the the the the the"),
    ("merhaba_loop", "Merhaba Merhaba Merhaba Merhaba Merhaba Merhaba"),
    ("bilmiyorum_loop", "Bilmiyorum Bilmiyorum Bilmiyorum Bilmiyorum Bilmiyorum"),
    ("ai_loop", "AI AI AI AI AI AI AI AI AI"),
)


class RuntimeLowInformationGuardTest(unittest.TestCase):
    def _assert_low_information_completion_is_fail_closed(self, raw_text: str) -> None:
        result = sanitize_completion(raw_text)

        self.assertEqual(result.text, "")
        self.assertIn("low_information_completion", result.warnings)
        self.assertIn("empty_after_output_guard", result.warnings)


def _make_low_information_test(raw_text: str):
    def test_case(self):
        self._assert_low_information_completion_is_fail_closed(raw_text)

    return test_case


for index, (name, raw_text) in enumerate(LOW_INFORMATION_CASES, start=1):
    setattr(
        RuntimeLowInformationGuardTest,
        f"test_low_information_case_{index:03d}_{name}",
        _make_low_information_test(raw_text),
    )


if __name__ == "__main__":
    unittest.main()
