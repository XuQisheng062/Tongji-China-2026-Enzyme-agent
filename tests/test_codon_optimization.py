from pathlib import Path

import pandas as pd

from fixed_enzyme_agent.codon_optimization import (
    fetch_kazusa_codon_profile,
    optimize_protein_to_cds,
    resolve_codon_optimization_request,
    search_kazusa_species,
    supported_hosts,
    write_codon_optimization_outputs,
)
from fixed_enzyme_agent.types import Candidate


def test_only_triggers_when_codon_optimization_is_requested():
    assert resolve_codon_optimization_request("\u76ee\u6807 pH 8.0\uff0c\u4f18\u5148\u6d3b\u6027")["enabled"] is False
    assert resolve_codon_optimization_request("\u8bf7\u505a\u5bc6\u7801\u5b50\u4f18\u5316")["enabled"] is True
    assert resolve_codon_optimization_request("Please perform codon optimization")["enabled"] is True


def test_default_host_is_ecoli_when_no_species_is_given():
    result = resolve_codon_optimization_request("\u8bf7\u5bf9\u6700\u7ec8\u5019\u9009\u505a\u5bc6\u7801\u5b50\u4f18\u5316")
    assert result["host_count"] == 1
    assert result["hosts"][0]["organism_key"] == "e_coli"
    assert result["selection_source"] == "default_e_coli"
    assert result["warning"]


def test_explicit_multiple_builtin_hosts_are_all_kept():
    result = resolve_codon_optimization_request(
        "\u8bf7\u5bf9\u6700\u7ec8\u5019\u9009\u5206\u522b\u505a\u5927\u80a0\u6746\u83cc\u3001\u67af\u8349\u82bd\u5b62\u6746\u83cc\u548c\u917f\u9152\u9175\u6bcd\u5bc6\u7801\u5b50\u4f18\u5316"
    )
    assert result["host_count"] == 3
    assert [host["organism_key"] for host in result["hosts"]] == [
        "e_coli",
        "b_subtilis",
        "s_cerevisiae",
    ]
    assert result["selection_source"] == "explicit_hosts"


def test_dynamic_latin_name_and_taxid_are_recorded_without_network():
    latin = resolve_codon_optimization_request(
        "Please perform codon optimization for Clostridium kluyveri"
    )
    assert latin["hosts"][0]["kind"] == "kazusa_name"
    assert latin["hosts"][0]["query"] == "Clostridium kluyveri"

    taxid = resolve_codon_optimization_request(
        "\u8bf7\u505a\u5bc6\u7801\u5b50\u4f18\u5316\uff0c\u8868\u8fbe\u5bbf\u4e3b\u4f7f\u7528 NCBI TaxID 12345"
    )
    assert taxid["hosts"][0]["kind"] == "kazusa_taxid"
    assert taxid["hosts"][0]["taxid"] == "12345"


def test_builtin_name_is_not_duplicated_as_dynamic_latin_name():
    result = resolve_codon_optimization_request(
        "codon optimization for Escherichia coli and Bacillus subtilis"
    )
    assert result["host_count"] == 2
    assert {host["organism_key"] for host in result["hosts"]} == {"e_coli", "b_subtilis"}


def test_supported_host_count_is_ten_plus_bsubtilis():
    hosts = supported_hosts()
    assert len(hosts) == 11
    keys = {host["key"] for host in hosts}
    assert "e_coli" in keys
    assert "b_subtilis" in keys


def test_ecoli_preferred_codon_back_translation():
    result = optimize_protein_to_cds("MKW", "e_coli")
    assert result["optimized_cds"] == "ATGAAATGG"
    assert result["optimized_cds_with_stop"] == "ATGAAATGGTAA"
    assert result["protein_length"] == 3
    assert result["cds_length"] == 9


def test_write_outputs_contains_each_sequence_for_each_host(tmp_path: Path):
    candidate = Candidate(mutation="K2R", wt_sequence="MKW")
    candidate.mutant_sequence = "MRW"
    resolved = resolve_codon_optimization_request(
        "\u8bf7\u4e3a\u5927\u80a0\u6746\u83cc\u548c\u67af\u8349\u82bd\u5b62\u6746\u83cc\u8fdb\u884c\u5bc6\u7801\u5b50\u4f18\u5316"
    )
    info = write_codon_optimization_outputs(
        run_dir=tmp_path,
        sequence_name="demo",
        wt_sequence="MKW",
        selected=[candidate],
        resolved_request=resolved,
        cache_dir=tmp_path / "cache",
    )
    assert info["status"] == "success"
    assert info["host_count"] == 2
    assert info["sequences_per_host"] == 2
    assert info["count"] == 4
    assert len(info["per_host_fastas"]) == 2

    csv_path = Path(info["csv"])
    fasta_path = Path(info["fasta"])
    assert csv_path.is_file()
    assert fasta_path.is_file()
    df = pd.read_csv(csv_path)
    assert set(df["mutation"]) == {"WT", "K2R"}
    assert set(df["organism_key"]) == {"e_coli", "b_subtilis"}
    assert len(df) == 4
    text = fasta_path.read_text(encoding="utf-8")
    assert "demo_WT__e_coli" in text
    assert "demo_WT__b_subtilis" in text


_CODON_TO_AA = {
    # Phe/Leu
    "UUU":"F","UUC":"F","UUA":"L","UUG":"L","CUU":"L","CUC":"L","CUA":"L","CUG":"L",
    # Ile/Met/Val
    "AUU":"I","AUC":"I","AUA":"I","AUG":"M","GUU":"V","GUC":"V","GUA":"V","GUG":"V",
    # Ser/Pro/Thr/Ala
    "UCU":"S","UCC":"S","UCA":"S","UCG":"S","CCU":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACU":"T","ACC":"T","ACA":"T","ACG":"T","GCU":"A","GCC":"A","GCA":"A","GCG":"A",
    # Tyr/stop/His/Gln/Asn/Lys/Asp/Glu
    "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*","CAU":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAU":"N","AAC":"N","AAA":"K","AAG":"K","GAU":"D","GAC":"D","GAA":"E","GAG":"E",
    # Cys/stop/Trp/Arg/Ser/Arg/Gly
    "UGU":"C","UGC":"C","UGA":"*","UGG":"W","CGU":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGU":"S","AGC":"S","AGA":"R","AGG":"R","GGU":"G","GGC":"G","GGA":"G","GGG":"G",
}


def _mock_kazusa_table_html(name: str = "Clostridium kluyveri", taxid: str = "12345") -> str:
    # Give lexicographically later codons a larger fraction within each synonymous group.
    grouped = {}
    for codon, aa in _CODON_TO_AA.items():
        grouped.setdefault(aa, []).append(codon)
    lines = []
    for aa, codons in grouped.items():
        for idx, codon in enumerate(sorted(codons), start=1):
            fraction = idx / sum(range(1, len(codons) + 1))
            lines.append(f"{codon} {aa} {fraction:.2f} {idx * 1.0:.1f} ( {idx * 10} )")
    return (
        f"<html><body><pre>{name} [gbbct]: 100 CDS's (10000 codons)\n"
        + "  ".join(lines)
        + "\nGenetic code 1: Standard</pre></body></html>"
    )


def test_kazusa_name_search_prefers_exact_match():
    html = '''
    <a href="showcodon.cgi?species=111">Clostridium kluyveri [gbbct]: 100</a>
    <a href="showcodon.cgi?species=222">Clostridium kluyveri DSMX [gbbct]: 50</a>
    '''

    def downloader(url, timeout):
        return url, html

    hit = search_kazusa_species("Clostridium kluyveri", downloader=downloader)
    assert hit == {"taxid": "111", "scientific_name": "Clostridium kluyveri"}


def test_dynamic_kazusa_profile_is_parsed_and_cached(tmp_path: Path):
    html = _mock_kazusa_table_html()
    calls = []

    def downloader(url, timeout):
        calls.append(url)
        return url, html

    profile = fetch_kazusa_codon_profile(
        "12345",
        scientific_name_hint="Clostridium kluyveri",
        cache_dir=tmp_path,
        downloader=downloader,
    )
    assert profile["taxid"] == "12345"
    assert profile["scientific_name"] == "Clostridium kluyveri"
    assert len(profile["preferred_codons"]) == 20
    assert profile["preferred_stop"] in {"TAA", "TAG", "TGA"}
    assert calls

    # Second load should come from cache and must not call the downloader.
    cached = fetch_kazusa_codon_profile(
        "12345",
        cache_dir=tmp_path,
        downloader=lambda *_: (_ for _ in ()).throw(AssertionError("network should not be used")),
    )
    assert cached["codon_source"] == "kazusa_cache"
