# app/ocr_enhancement.py
# YENİ DOSYA - Complete OCR accuracy enhancement utilities

import re
import logging

logger = logging.getLogger(__name__)

# Province codes (01-81) for validation
VALID_PROVINCE_CODES = {
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
    '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
    '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
    '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81'
}

# Position-aware OCR corrections
LETTER_TO_NUMBER = {'B': '8', 'D': '0', 'G': '6', 'S': '5', 'O': '0', 'I': '1', 'Z': '2'}
NUMBER_TO_LETTER = {'8': 'B', '0': 'O', '6': 'G', '5': 'S', '1': 'I', '2': 'Z'}

def enhanced_clean_text(text):
    """
    Enhanced text cleaning with smart context-aware OCR error correction
    """
    if not text:
        return ""
    
    # Basic cleanup
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    
    # Apply context-aware corrections
    text = context_aware_correction(text)
    
    # Length check
    if len(text) < 2 or len(text) > 10:
        return ""
    
    return text

def enhanced_validation(text):
    """
    Enhanced plate format validation supporting all Turkish plate combinations
    """
    if not text or len(text) < 5:
        return False
    
    # Standard Turkish format: 34[1-3 letters][1-4 numbers]
    # All valid combinations:
    valid_patterns = [
        # 1 letter + 1-4 numbers
        r'^[0-9]{2}[A-Z]{1}[0-9]{1}$',   # 34A1
        r'^[0-9]{2}[A-Z]{1}[0-9]{2}$',   # 34A12
        r'^[0-9]{2}[A-Z]{1}[0-9]{3}$',   # 34A123
        r'^[0-9]{2}[A-Z]{1}[0-9]{4}$',   # 34A1234
        
        # 2 letters + 1-4 numbers
        r'^[0-9]{2}[A-Z]{2}[0-9]{1}$',   # 34AB1
        r'^[0-9]{2}[A-Z]{2}[0-9]{2}$',   # 34AB12
        r'^[0-9]{2}[A-Z]{2}[0-9]{3}$',   # 34AB123
        r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',   # 34AB1234
        
        # 3 letters + 1-4 numbers
        r'^[0-9]{2}[A-Z]{3}[0-9]{1}$',   # 34ABC1
        r'^[0-9]{2}[A-Z]{3}[0-9]{2}$',   # 34ABC12
        r'^[0-9]{2}[A-Z]{3}[0-9]{3}$',   # 34ABC123
        r'^[0-9]{2}[A-Z]{3}[0-9]{4}$',   # 34ABC1234
    ]
    
    for pattern in valid_patterns:
        if re.match(pattern, text):
            # Check if province code is valid
            province = text[:2]
            if province in VALID_PROVINCE_CODES:
                return True
    
    # Diplomatic format: ABC1234 or ABC1234D
    if re.match(r'^[A-Z]{2,3}[0-9]{3,4}[A-Z]?$', text):
        return True
    
    # Old format: A1234BC
    if re.match(r'^[A-Z]{1,2}[0-9]{2,4}[A-Z]{1,2}$', text):
        return True
    
    return False

def context_aware_correction(text):
    """
    Context-aware correction that analyzes the entire plate to make better decisions
    """
    if not text or len(text) < 5:
        return text
    
    # Remove any spaces or special characters
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Analyze the pattern to determine the most likely structure
    analysis = analyze_plate_pattern(clean_text)
    
    if analysis['confidence'] > 0.6:
        return apply_corrections_based_on_analysis(clean_text, analysis)
    else:
        # Fallback to conservative correction
        return apply_conservative_corrections(clean_text)

def analyze_plate_pattern(text):
    """
    Analyze the text to determine the most likely plate structure
    Supports all Turkish plate combinations: 1-3 letters + 1-4 numbers
    """
    results = []
    
    # All possible Turkish plate patterns after province code (34...)
    standard_patterns = [
        # 1 letter + 1-4 numbers
        {'regex': r'^([0-9]{2})([A-Z]{1})([0-9]{1})$', 'letters': 1, 'numbers': 1, 'confidence': 0.7},
        {'regex': r'^([0-9]{2})([A-Z]{1})([0-9]{2})$', 'letters': 1, 'numbers': 2, 'confidence': 0.8},
        {'regex': r'^([0-9]{2})([A-Z]{1})([0-9]{3})$', 'letters': 1, 'numbers': 3, 'confidence': 0.9},
        {'regex': r'^([0-9]{2})([A-Z]{1})([0-9]{4})$', 'letters': 1, 'numbers': 4, 'confidence': 0.9},
        
        # 2 letters + 1-4 numbers  
        {'regex': r'^([0-9]{2})([A-Z]{2})([0-9]{1})$', 'letters': 2, 'numbers': 1, 'confidence': 0.8},
        {'regex': r'^([0-9]{2})([A-Z]{2})([0-9]{2})$', 'letters': 2, 'numbers': 2, 'confidence': 0.9},
        {'regex': r'^([0-9]{2})([A-Z]{2})([0-9]{3})$', 'letters': 2, 'numbers': 3, 'confidence': 0.9},
        {'regex': r'^([0-9]{2})([A-Z]{2})([0-9]{4})$', 'letters': 2, 'numbers': 4, 'confidence': 0.8},
        
        # 3 letters + 1-4 numbers
        {'regex': r'^([0-9]{2})([A-Z]{3})([0-9]{1})$', 'letters': 3, 'numbers': 1, 'confidence': 0.9},
        {'regex': r'^([0-9]{2})([A-Z]{3})([0-9]{2})$', 'letters': 3, 'numbers': 2, 'confidence': 0.8},
        {'regex': r'^([0-9]{2})([A-Z]{3})([0-9]{3})$', 'letters': 3, 'numbers': 3, 'confidence': 0.7},
        {'regex': r'^([0-9]{2})([A-Z]{3})([0-9]{4})$', 'letters': 3, 'numbers': 4, 'confidence': 0.6},
    ]
    
    # Test each pattern against original text
    for pattern in standard_patterns:
        match = re.match(pattern['regex'], text)
        if match:
            groups = match.groups()
            province, letters, numbers = groups
            
            score = 0.5
            
            # Province validation
            if province in VALID_PROVINCE_CODES:
                score += 0.3
            
            # Pattern likelihood scoring
            if pattern['letters'] == 2 and 2 <= pattern['numbers'] <= 3:
                score += 0.2  # Most common: 34AB123, 34AB12
            elif pattern['letters'] == 3 and pattern['numbers'] == 1:
                score += 0.15  # Common: 34ABC1
            elif pattern['letters'] == 1 and pattern['numbers'] == 4:
                score += 0.15  # Common: 34A1234
            
            results.append({
                'type': 'standard',
                'groups': groups,
                'confidence': score * pattern['confidence'],
                'pattern': pattern,
                'expected_letters': pattern['letters'],
                'expected_numbers': pattern['numbers']
            })
    
    # Test with intelligent corrections applied
    corrected_variants = generate_correction_variants(text)
    for variant in corrected_variants:
        for pattern in standard_patterns:
            match = re.match(pattern['regex'], variant['text'])
            if match:
                groups = match.groups()
                province, letters, numbers = groups
                
                score = 0.4  # Base score for corrected text
                
                if province in VALID_PROVINCE_CODES:
                    score += 0.3
                
                score += variant['correction_score']
                
                results.append({
                    'type': 'standard',
                    'groups': groups,
                    'confidence': score * pattern['confidence'] * 0.9,
                    'pattern': pattern,
                    'expected_letters': pattern['letters'],
                    'expected_numbers': pattern['numbers'],
                    'corrected_text': variant['text'],
                    'corrections_made': variant['corrections']
                })
    
    # Add special formats
    results.extend(analyze_special_formats(text))
    
    # Return best analysis
    if results:
        best = max(results, key=lambda x: x['confidence'])
        logger.debug(f"Pattern analysis: {best.get('expected_letters', '?')}L+{best.get('expected_numbers', '?')}N, confidence={best['confidence']:.2f}")
        return best
    
    return {'confidence': 0.0, 'type': 'unknown'}

def generate_correction_variants(text):
    """
    Generate intelligent correction variants to test different possibilities
    """
    variants = []
    
    if len(text) < 5:
        return variants
    
    # Strategy 1: Conservative province correction only
    province_corrected = text[:2]
    for letter, number in {'O': '0', 'I': '1', 'D': '0'}.items():
        province_corrected = province_corrected.replace(letter, number)
    
    if province_corrected != text[:2]:
        variants.append({
            'text': province_corrected + text[2:],
            'correction_score': 0.1,
            'corrections': [f"Province: {text[:2]} → {province_corrected}"]
        })
    
    # Strategy 2: Try different letter/number boundary assumptions
    remaining = text[2:]  # After province
    
    # Test all possible split points
    for split_point in range(1, len(remaining)):
        potential_letters = remaining[:split_point]
        potential_numbers = remaining[split_point:]
        
        # Don't test if lengths are outside valid ranges
        if not (1 <= len(potential_letters) <= 3 and 1 <= len(potential_numbers) <= 4):
            continue
        
        # Apply corrections based on assumed positions
        corrected_letters = potential_letters
        for number, letter in NUMBER_TO_LETTER.items():
            corrected_letters = corrected_letters.replace(number, letter)
        
        corrected_numbers = potential_numbers
        # Conservative number corrections (avoid G→6 problem)
        conservative_fixes = {'O': '0', 'I': '1', 'S': '5', 'Z': '2'}
        for letter, number in conservative_fixes.items():
            corrected_numbers = corrected_numbers.replace(letter, number)
        
        corrected_variant = province_corrected + corrected_letters + corrected_numbers
        
        if corrected_variant != text:
            score = calculate_variant_score(potential_letters, potential_numbers, corrected_letters, corrected_numbers)
            
            variants.append({
                'text': corrected_variant,
                'correction_score': score,
                'corrections': [
                    f"Letters: {potential_letters} → {corrected_letters}",
                    f"Numbers: {potential_numbers} → {corrected_numbers}"
                ]
            })
    
    return variants

def calculate_variant_score(orig_letters, orig_numbers, corr_letters, corr_numbers):
    """
    Calculate quality score for a correction variant
    """
    score = 0.0
    
    # Bonus for letter section being more letter-like after correction
    if corr_letters.isalpha() and not orig_letters.isalpha():
        score += 0.15
    
    # Bonus for number section being more number-like after correction
    if corr_numbers.isdigit() and not orig_numbers.isdigit():
        score += 0.15
    
    # Penalty for too many corrections
    letter_changes = sum(1 for i, c in enumerate(orig_letters) if i < len(corr_letters) and c != corr_letters[i])
    number_changes = sum(1 for i, c in enumerate(orig_numbers) if i < len(corr_numbers) and c != corr_numbers[i])
    
    total_changes = letter_changes + number_changes
    if total_changes <= 2:
        score += 0.1
    elif total_changes <= 4:
        score += 0.05
    else:
        score -= 0.1
    
    return score

def analyze_special_formats(text):
    """
    Analyze special format plates (diplomatic, old format, etc.)
    """
    results = []
    
    # Diplomatic format: ABC1234 or ABC1234D
    diplomatic_patterns = [
        {'regex': r'^([A-Z]{2,3})([0-9]{3,4})([A-Z]?)$', 'type': 'diplomatic', 'confidence': 0.8}
    ]
    
    for pattern in diplomatic_patterns:
        match = re.match(pattern['regex'], text)
        if match:
            results.append({
                'type': 'diplomatic',
                'groups': match.groups(),
                'confidence': pattern['confidence'],
                'pattern': pattern
            })
    
    # Old format: A1234BC
    old_patterns = [
        {'regex': r'^([A-Z]{1,2})([0-9]{2,4})([A-Z]{1,2})$', 'type': 'old', 'confidence': 0.6}
    ]
    
    for pattern in old_patterns:
        match = re.match(pattern['regex'], text)
        if match:
            results.append({
                'type': 'old',
                'groups': match.groups(),
                'confidence': pattern['confidence'],
                'pattern': pattern
            })
    
    return results

def apply_corrections_based_on_analysis(text, analysis):
    """
    Apply corrections based on the structural analysis with full pattern support
    """
    if analysis['type'] == 'standard':
        return apply_dynamic_standard_corrections(text, analysis)
    elif analysis['type'] == 'diplomatic':
        return apply_diplomatic_corrections(text, analysis)
    elif analysis['type'] == 'old':
        return apply_old_format_corrections(text, analysis)
    else:
        return text

def apply_dynamic_standard_corrections(text, analysis):
    """
    Apply corrections for standard format with dynamic letter/number boundaries
    """
    if 'corrected_text' in analysis:
        result = analysis['corrected_text']
        logger.debug(f"Dynamic correction ({analysis['expected_letters']}L+{analysis['expected_numbers']}N): {text} → {result}")
        return result
    
    # If no pre-corrected text, apply basic corrections
    groups = analysis['groups']
    if len(groups) >= 3:
        province, letters, numbers = groups[:3]
        
        # Fix province
        fixed_province = province
        for letter, number in LETTER_TO_NUMBER.items():
            fixed_province = fixed_province.replace(letter, number)
        
        result = fixed_province + letters + numbers
        
        if result != text:
            logger.debug(f"Basic correction ({analysis['expected_letters']}L+{analysis['expected_numbers']}N): {text} → {result}")
        
        return result
    
    return text

def apply_diplomatic_corrections(text, analysis):
    """
    Apply corrections for diplomatic format plates
    """
    groups = analysis['groups']
    if len(groups) >= 2:
        letters, numbers = groups[:2]
        
        # Conservative corrections for diplomatic plates
        fixed_letters = letters
        for number, letter in {'0': 'O', '1': 'I'}.items():
            fixed_letters = fixed_letters.replace(number, letter)
        
        fixed_numbers = numbers
        for letter, number in {'O': '0', 'I': '1', 'S': '5'}.items():
            fixed_numbers = fixed_numbers.replace(letter, number)
        
        result = fixed_letters + fixed_numbers
        if len(groups) > 2 and groups[2]:  # Trailing letter
            result += groups[2]
        
        return result
    
    return text

def apply_old_format_corrections(text, analysis):
    """
    Apply corrections for old format plates: A1234BC
    """
    groups = analysis['groups']
    if len(groups) >= 3:
        letters1, numbers, letters2 = groups
        
        # Conservative corrections
        fixed_letters1 = letters1
        for number, letter in {'0': 'O', '1': 'I'}.items():
            fixed_letters1 = fixed_letters1.replace(number, letter)
        
        fixed_numbers = numbers
        for letter, number in {'O': '0', 'I': '1', 'S': '5'}.items():
            fixed_numbers = fixed_numbers.replace(letter, number)
        
        fixed_letters2 = letters2
        for number, letter in {'0': 'O', '1': 'I'}.items():
            fixed_letters2 = fixed_letters2.replace(number, letter)
        
        return fixed_letters1 + fixed_numbers + fixed_letters2
    
    return text

def apply_conservative_corrections(text):
    """
    Apply very conservative corrections as fallback
    """
    corrected = text
    
    # Only fix very obvious confusions
    obvious_fixes = {'O': '0', 'I': '1'}
    
    for old, new in obvious_fixes.items():
        corrected = corrected.replace(old, new)
    
    return corrected

def calculate_confidence_boost(text, original_confidence):
    """
    Calculate confidence boost based on format validity and corrections made
    """
    if enhanced_validation(text):
        # Check if it's a perfect Turkish standard format
        if re.match(r'^[0-9]{2}[A-Z]{1,3}[0-9]{1,4}$', text):
            province = text[:2]
            if province in VALID_PROVINCE_CODES:
                return min(1.0, original_confidence * 1.3)  # 30% boost for perfect format
        
        return min(1.0, original_confidence * 1.15)  # 15% boost for valid format
    
    return max(0.1, original_confidence * 0.9)  # Slight penalty for invalid format

def smart_ensemble_decision(easyocr_text, easyocr_conf, paddleocr_text, paddleocr_conf):
    """
    Smart ensemble decision with position-aware validation for all combinations
    """
    # Clean both texts
    easy_cleaned = enhanced_clean_text(easyocr_text)
    paddle_cleaned = enhanced_clean_text(paddleocr_text)
    
    # Validate both
    easy_valid = enhanced_validation(easy_cleaned)
    paddle_valid = enhanced_validation(paddle_cleaned)
    
    # Decision logic
    if easy_cleaned == paddle_cleaned and easy_cleaned:
        # Both engines agree after cleaning
        confidence = max(easyocr_conf, paddleocr_conf)
        return easy_cleaned, calculate_confidence_boost(easy_cleaned, confidence), "both_agree"
    
    elif easy_valid and not paddle_valid:
        # Only EasyOCR gives valid format
        return easy_cleaned, calculate_confidence_boost(easy_cleaned, easyocr_conf), "easyocr_valid"
    
    elif paddle_valid and not easy_valid:
        # Only PaddleOCR gives valid format
        return paddle_cleaned, calculate_confidence_boost(paddle_cleaned, paddleocr_conf), "paddleocr_valid"
    
    elif easy_valid and paddle_valid:
        # Both valid, choose higher confidence
        if easyocr_conf >= paddleocr_conf:
            return easy_cleaned, calculate_confidence_boost(easy_cleaned, easyocr_conf), "both_valid_easy_higher"
        else:
            return paddle_cleaned, calculate_confidence_boost(paddle_cleaned, paddleocr_conf), "both_valid_paddle_higher"
    
    else:
        # Neither valid, choose higher confidence
        if easyocr_conf >= paddleocr_conf:
            return easy_cleaned, easyocr_conf, "neither_valid_easy_higher"
        else:
            return paddle_cleaned, paddleocr_conf, "neither_valid_paddle_higher"