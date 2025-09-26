#!/usr/bin/env python3
"""
Test script to reproduce the 404 connection error
"""

import os
import sys
import traceback
from openai import OpenAI

def test_openai_connection():
    """Test OpenAI connection with different scenarios"""
    print("🔍 Testing OpenAI connection...")

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False

    print(f"✅ API key found (length: {len(api_key)})")

    try:
        # Test 1: Basic client creation
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")

        # Test 2: Simple chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは建機の修理データアナリストです。"},
                {"role": "user", "content": "テスト"}
            ],
            temperature=0.1,
            max_tokens=50,
            timeout=30
        )
        print("✅ Basic chat completion successful")
        print(f"Response: {response.choices[0].message.content}")

        # Test 3: Longer prompt (similar to app usage)
        system_prompt = """あなたは建機の修理データアナリストです。
与えられた日本語の質問を、pandasで実行可能な集計仕様のJSONに変換してください。
データ列名は日本語です。利用可能な列名: 機種名, 作業内容, 不具合内容

出力JSONスキーマ：
{
  "time_range": {"from": "YYYY-MM-DD|null", "to": "YYYY-MM-DD|null"},
  "filters": [{"column": "列名", "op": "==|!=|in|not in|contains|regex|fuzzy_search", "value": "any"}, ...],
  "groupby": ["列名", ...],
  "metrics": [{"name":"件数","expr":"count"}, {"name":"数量合計","expr":"sum:部品数量"}, ...],
  "topn": 10,
  "sort": {"by":"件数","asc":false}
}

JSONのみを出力してください。説明は不要です。"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "エンジン関連の故障件数を教えて"}
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=30
        )
        print("✅ Complex chat completion successful")
        print(f"Response length: {len(response.choices[0].message.content)}")

        return True

    except Exception as e:
        print(f"❌ Error occurred: {type(e).__name__}: {str(e)}")
        if "404" in str(e):
            print("   This is a 404 error - model or endpoint not found")
        elif "401" in str(e):
            print("   This is a 401 error - authentication failed")
        elif "timeout" in str(e).lower():
            print("   This is a timeout error")

        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_from_kubotad_code():
    """Test using the exact same code as KUBOTAD.py"""
    print("\n🔍 Testing with KUBOTAD.py code...")

    try:
        # Import the exact function from KUBOTAD
        sys.path.append('/workspaces/AI_APP_Repository')
        from KUBOTAD import llm_to_json_spec

        available_columns = ['機種名', '作業内容', '不具合内容', '部品品名', 'アワメータ_HR']
        question = "エンジン関連の故障件数"

        result = llm_to_json_spec(question, available_columns)

        if result:
            print("✅ KUBOTAD function test successful")
            print(f"Result: {result}")
        else:
            print("❌ KUBOTAD function returned empty result")

    except Exception as e:
        print(f"❌ KUBOTAD function test failed: {type(e).__name__}: {str(e)}")
        if "404" in str(e):
            print("   Found the 404 error in KUBOTAD code!")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting connection tests...\n")

    # Test basic connection
    basic_test = test_openai_connection()

    # Test KUBOTAD specific code
    test_from_kubotad_code()

    print(f"\n📊 Test Results:")
    print(f"   Basic OpenAI: {'✅ PASS' if basic_test else '❌ FAIL'}")