"""Tests for allocator/ibkr_data.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from ibkr_data import parse_report_snapshot


def test_parse_report_snapshot_extracts_core_ratios():
    xml_text = """
    <ReportSnapshot>
      <Ratios>
        <Group>
          <Ratio FieldName="PEEXCLXOR">12.5</Ratio>
          <Ratio FieldName="PRICE2BK">2.1</Ratio>
          <Ratio FieldName="TTMEPSXCLX">4.2</Ratio>
          <Ratio FieldName="MKTCAP">1000.0</Ratio>
          <Ratio FieldName="TTMREV">200.0</Ratio>
          <Ratio FieldName="TTMNIAC">50.0</Ratio>
        </Group>
      </Ratios>
      <ForecastData>
        <Ratio FieldName="ProjLTGrowthRate" Type="N">
          <Value PeriodType="CURR">10.0</Value>
        </Ratio>
        <Ratio FieldName="ProjPE" Type="N">
          <Value PeriodType="CURR">15.0</Value>
        </Ratio>
      </ForecastData>
    </ReportSnapshot>
    """
    parsed = parse_report_snapshot(xml_text)
    assert parsed["trailing_pe"] == 12.5
    assert parsed["forward_pe"] == 15.0
    assert parsed["price_to_book"] == 2.1
    assert parsed["earnings_growth_5y"] == 0.1
    assert parsed["peg_ratio"] == 1.5
