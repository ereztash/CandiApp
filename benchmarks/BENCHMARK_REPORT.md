# CandiApp Resume Parser - Benchmark Report

**Date:** November 17, 2025
**Version:** 0.1.0 (MVP)
**Dataset:** Synthetic Resume Dataset (50 resumes)
**Benchmark Framework:** Based on RChilli 7-Criteria Framework

---

## Executive Summary

CandiApp Resume Parser was benchmarked against a synthetic dataset of 50 professional resumes. The parser demonstrates **excellent speed performance** but requires **significant accuracy improvements** to meet industry standards.

### Overall Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Parsing Speed** | 0.86ms avg | <1000ms | ✅ **PASS** |
| **Parsing Accuracy** | 33.3% | >90% | ❌ **FAIL** |
| **Fields Extracted** | 19 avg | >50 | ❌ **FAIL** |
| **Error Rate** | 0% | <5% | ✅ **PASS** |

**Overall Status:** ⚠️ **PARTIAL PASS** (2/4 criteria met)

---

## 1. Parsing Speed Performance

### Results

```
Average Time:     0.86ms (0.0008566s)
Median Time:      0.84ms
Min/Max Time:     0.66ms / 1.49ms
Industry Target:  <1000ms (1 second)
```

### Analysis

✅ **EXCELLENT PERFORMANCE**

The parser significantly **exceeds industry standards**:
- **1,167x faster** than the 1-second industry target
- **350x faster** than Textkernel's 300ms benchmark
- Consistent performance across all 50 resumes

### Comparison to Industry Benchmarks

| Parser | Speed | CandiApp vs Industry |
|--------|-------|----------------------|
| **CandiApp** | **0.86ms** | **Baseline** |
| Textkernel | 300ms | 350x slower |
| RChilli | ~20ms | 23x slower |
| Industry Target | <1000ms | 1,167x slower |

**Verdict:** Speed performance is **production-ready** and **industry-leading**.

---

## 2. Parsing Accuracy

### Results by Field

| Field | Accuracy | Target | Status |
|-------|----------|--------|--------|
| **Name** | 100.0% | >90% | ✅ PASS |
| **Email** | 100.0% | >90% | ✅ PASS |
| **Phone** | 0.0% | >90% | ❌ FAIL |
| **Skills** | 0.0% | >90% | ❌ FAIL |
| **Experience Count** | 0.0% | >80% | ❌ FAIL |
| **Education Count** | 0.0% | >80% | ❌ FAIL |
| **Overall** | **33.3%** | **>90%** | ❌ **FAIL** |

### Analysis

❌ **NEEDS SIGNIFICANT IMPROVEMENT**

**Strengths:**
- Perfect name extraction (100%)
- Perfect email extraction (100%)
- Robust regex-based field detection

**Critical Gaps:**
- Phone number parsing not working (regex pattern issue)
- Skills extraction not implemented (returns empty lists)
- Experience parsing not implemented (empty array)
- Education parsing not implemented (empty array)

### Root Cause Analysis

The current parser implementation (`src/candiapp/parser.py`) has:

1. **Implemented:**
   - ✅ Name extraction (basic heuristic)
   - ✅ Email extraction (regex)
   - ✅ Contact info structure

2. **Partially Implemented:**
   - ⚠️ Phone extraction (regex exists but not matching format)
   - ⚠️ Summary extraction (basic pattern matching)

3. **Not Implemented:**
   - ❌ Skills extraction (returns empty list)
   - ❌ Experience extraction (returns empty list)
   - ❌ Education extraction (returns empty list)
   - ❌ Certifications extraction
   - ❌ Languages extraction

---

## 3. Field Extraction

### Results

```
Average Fields Extracted:  19 fields
MVP Target:                50 fields
Production Target:         200+ fields
```

### Analysis

❌ **BELOW TARGET**

Currently extracting only **38%** of MVP target fields:
- Contact information: 5 fields
- Personal information: 3 fields
- Metadata: 11 fields

**Missing critical fields:**
- Work experience entries (0)
- Education entries (0)
- Skills list (0)
- Certifications (0)
- Languages (0)
- Projects (0)
- Awards/Publications (0)

---

## 4. Error Handling

### Results

```
Total Resumes:    50
Parsing Errors:   0
Error Rate:       0.0%
Target:           <5%
```

### Analysis

✅ **EXCELLENT**

- No parsing failures across all 50 resumes
- Robust error handling
- All files processed successfully

---

## 5. Comparison to Industry Benchmarks

### Industry Leaders (from docs/benchmarks.md)

| Solution | Speed | Accuracy | Fields | Our Performance |
|----------|-------|----------|--------|-----------------|
| **Textkernel** | 300ms | 85-92% | 200+ | Speed: ✅ (350x faster)<br>Accuracy: ❌ (33% vs 88%)<br>Fields: ❌ (19 vs 200) |
| **RChilli** | 20ms | 85-92% | 200+ | Speed: ✅ (23x faster)<br>Accuracy: ❌ (33% vs 88%)<br>Fields: ❌ (19 vs 200) |
| **Fabric** | Variable | ~90% | 200+ | Speed: ✅<br>Accuracy: ❌ (33% vs 90%)<br>Fields: ❌ (19 vs 200) |
| **Kickresume** | Variable | 92% | Variable | Speed: ✅<br>Accuracy: ❌ (33% vs 92%) |

### Our Benchmark Targets (from docs/benchmarks.md)

| Milestone | Speed | Accuracy | Fields | Status |
|-----------|-------|----------|--------|--------|
| **MVP** | <2s | >85% | 50+ | Speed: ✅<br>Accuracy: ❌<br>Fields: ❌ |
| **v1.0 (Production)** | <1s | >90% | 100+ | Speed: ✅<br>Accuracy: ❌<br>Fields: ❌ |
| **v2.0 (Industry-Leading)** | <500ms | >92% | 200+ | Speed: ✅<br>Accuracy: ❌<br>Fields: ❌ |

---

## 6. Key Findings

### ✅ Strengths

1. **Exceptional Speed**: 0.86ms average (1,167x faster than industry target)
2. **Perfect Name Extraction**: 100% accuracy
3. **Perfect Email Extraction**: 100% accuracy
4. **Zero Errors**: 100% success rate, no parsing failures
5. **Production-Ready Infrastructure**: Clean code, proper error handling

### ❌ Critical Gaps

1. **Skills Extraction**: 0% (not implemented)
2. **Experience Parsing**: 0% (not implemented)
3. **Education Parsing**: 0% (not implemented)
4. **Phone Number Parsing**: 0% (regex issue)
5. **Overall Accuracy**: 33.3% (far below 90% target)
6. **Field Count**: 19 fields (62% below MVP target of 50)

---

## 7. Recommendations

### Immediate Priority (MVP Completion)

1. **Implement Skills Extraction** (Priority: CRITICAL)
   - Use keyword matching against skill database
   - Implement section detection (SKILLS, TECHNICAL SKILLS headers)
   - Add common tech skills taxonomy

2. **Implement Experience Parsing** (Priority: CRITICAL)
   - Detect EXPERIENCE section
   - Extract company names, job titles, dates
   - Parse bullet points for responsibilities

3. **Implement Education Parsing** (Priority: CRITICAL)
   - Detect EDUCATION section
   - Extract university names, degrees, graduation dates
   - Identify education levels (Bachelor, Master, PhD)

4. **Fix Phone Number Regex** (Priority: HIGH)
   - Test regex against actual phone formats in dataset
   - Support multiple phone number formats
   - Normalize phone numbers for comparison

5. **Implement Section Detection** (Priority: HIGH)
   - Create robust header detection (case-insensitive)
   - Support multiple languages (English, Hebrew)
   - Handle various formatting styles

### Short-Term (v1.0 - Production Ready)

6. **Add NLP-Based Parsing**
   - Integrate spaCy for entity recognition
   - Implement named entity recognition (NER) for names, companies
   - Use NLP for date extraction

7. **Improve Field Extraction**
   - Increase from 19 to 100+ fields
   - Add certifications, languages, projects
   - Extract detailed work responsibilities

8. **Enhance Accuracy**
   - Target 90%+ overall accuracy
   - Implement validation for extracted fields
   - Add confidence scores

### Long-Term (v2.0 - Industry-Leading)

9. **Semantic Matching**
   - Implement contextual understanding
   - Add transferable skills detection
   - Support skill synonyms and variations

10. **Multi-Language Support**
    - Full Hebrew parsing support
    - Support RTL text handling
    - Multi-language skill taxonomies

11. **Advanced Features**
    - Career progression analysis
    - Seniority level detection
    - Industry classification

---

## 8. Next Steps

### Development Roadmap

**Phase 1: MVP Completion** (Est. 2-3 weeks)
- [ ] Implement skills extraction
- [ ] Implement experience parsing
- [ ] Implement education parsing
- [ ] Fix phone number regex
- [ ] Achieve >85% accuracy
- [ ] Extract 50+ fields minimum

**Phase 2: Production Ready** (Est. 1-2 months)
- [ ] Add NLP support (spaCy integration)
- [ ] Reach 90%+ accuracy
- [ ] Extract 100+ fields
- [ ] Add certifications, languages, projects
- [ ] Comprehensive testing on real resumes

**Phase 3: Industry-Leading** (Est. 3-6 months)
- [ ] Semantic matching implementation
- [ ] Full Hebrew support
- [ ] 200+ field extraction
- [ ] 92%+ accuracy
- [ ] Advanced ML features

---

## 9. Conclusion

CandiApp Resume Parser shows **exceptional promise** with industry-leading speed performance but requires **substantial feature development** to reach production readiness.

**Current State:**
- ✅ Speed: Production-ready (exceeds industry standards by 350-1,167x)
- ⚠️ Accuracy: Development stage (33% vs 90% target)
- ❌ Features: Basic implementation (19 vs 200+ fields)

**Path to Production:**
The parser has a **solid foundation** with excellent performance and zero errors. Implementing the critical missing features (skills, experience, education extraction) would bring the system to MVP status within 2-3 weeks. The modular architecture supports rapid feature addition.

**Recommendation:**
- **Continue development** with focus on high-priority features
- **Re-benchmark** after MVP features are implemented
- **Target** achieving 85%+ accuracy and 50+ fields for initial release

---

## 10. Technical Details

### Test Environment
- **Platform:** Linux 4.4.0
- **Python Version:** 3.x
- **Dataset:** 50 synthetic resumes (TXT format)
- **Parser Configuration:** NLP disabled (enable_nlp=False)
- **Benchmark Date:** November 17, 2025

### Dataset Characteristics
- Total resumes: 50
- Average experience: 7.6 years
- Average skills per resume: 8.3
- Total experiences: 169
- Total education entries: 81
- Languages: English (with some Hebrew names)

### Benchmark Methodology
Based on RChilli's 7-Criteria Framework:
1. ✅ Parsing Speed
2. ❌ Parsing Accuracy
3. ❌ Number of Data Fields
4. ⚠️ Language Support
5. N/A Taxonomy Richness
6. N/A Data Security
7. N/A Integration Capability

---

**Generated:** November 17, 2025
**Tool:** CandiApp Benchmark Suite v0.1.0
**Contact:** info@candiapp.dev
