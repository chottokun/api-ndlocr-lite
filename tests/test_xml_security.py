import pytest
from defusedxml import EntitiesForbidden
from src.core.engine import ET

def test_xml_entity_expansion_forbidden():
    """
    Test that the parser in src.core.engine (which should be defusedxml)
    blocks entity expansion (Billion Laughs attack).
    """
    malicious_xml = """<?xml version="1.0"?>
    <!DOCTYPE lolz [
     <!ENTITY lol "lol">
     <!ELEMENT lolz (#PCDATA)>
     <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
     <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
     <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
    ]>
    <lolz>&lol3;</lolz>
    """

    with pytest.raises(EntitiesForbidden):
        ET.fromstring(malicious_xml)

def test_xml_external_entity_forbidden():
    """
    Test that the parser blocks external entities (XXE).
    """
    malicious_xml = """<?xml version="1.0"?>
    <!DOCTYPE data [
      <!ENTITY xxe SYSTEM "file:///etc/passwd">
    ]>
    <data>&xxe;</data>
    """

    with pytest.raises(EntitiesForbidden):
        ET.fromstring(malicious_xml)
