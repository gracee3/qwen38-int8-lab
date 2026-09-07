
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quant'))
from calibration.prepare_int4_corpus import coherent_prefixes, messages_for


class ConversationIntegrity(unittest.TestCase):
    def test_tool_call_prefix_includes_matching_result(self):
        messages=[{'role':'user','content':'task'}, {'role':'assistant','tool_calls':[{'id':'a','function':{'name':'shell','arguments':{'command':'pwd'}}}]}, {'role':'tool','tool_call_id':'a','content':'/tmp'}]
        prefixes=list(coherent_prefixes(messages))
        self.assertEqual(prefixes,[messages])

    def test_incomplete_parallel_tools_are_not_selected(self):
        messages=[{'role':'assistant','tool_calls':[{'id':'a'},{'id':'b'}]}, {'role':'tool','tool_call_id':'a','content':'result'}]
        self.assertEqual(list(coherent_prefixes(messages)),[])

    def test_orphan_tool_is_rejected(self):
        self.assertEqual(list(coherent_prefixes([{'role':'tool','tool_call_id':'missing','content':'result'}])),[])

    def test_separate_reasoning_excluded_and_tool_fields_preserved(self):
        call={'id':'a','function':{'name':'shell','arguments':{'command':'pwd'}}}
        row={'reasoning':'private-field','messages':[{'role':'assistant','content':'','tool_calls':[call],'reasoning_content':'separate-field'}]}
        messages=messages_for(row)
        self.assertEqual(messages,[{'role':'assistant','content':'','tool_calls':[call]}])

    def test_rust_uses_content_field(self):
        self.assertEqual(messages_for({'input_data':'task','output_data':'answer'}),[{'role':'user','content':'task'},{'role':'assistant','content':'answer'}])

    def test_json_tool_arguments_normalized_without_changing_values(self):
        messages=messages_for({'messages':[{'role':'assistant','tool_calls':[{'id':'a','function':{'name':'shell','arguments':'{"command":"pwd"}'}}]}]})
        self.assertEqual(messages[0]['tool_calls'][0]['function']['arguments'],{'command':'pwd'})

    def test_non_mapping_tool_arguments_rejected(self):
        self.assertEqual(messages_for({'messages':[{'role':'assistant','tool_calls':[{'id':'a','function':{'arguments':'[]'}}]}]}),[])


if __name__=='__main__': unittest.main()
