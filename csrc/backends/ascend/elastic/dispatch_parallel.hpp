#pragma once

#include <cstdint>
#include <limits>

namespace deep_ep::ascend::elastic {

inline constexpr std::uint64_t kDispatchBitmapWordBits = 64;

constexpr bool dispatch_bitmap_words(
    std::uint64_t bits, std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    *words = bits / kDispatchBitmapWordBits +
        (bits % kDispatchBitmapWordBits != 0 ? 1 : 0);
    return true;
}

constexpr bool dispatch_owner_bitmap_words(
    std::uint64_t owners, std::uint64_t bits_per_owner,
    std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    std::uint64_t words_per_owner = 0;
    if (!dispatch_bitmap_words(bits_per_owner, &words_per_owner) ||
        (words_per_owner != 0 &&
         owners > std::numeric_limits<std::uint64_t>::max() /
             words_per_owner))
        return false;
    *words = owners * words_per_owner;
    return true;
}

constexpr bool dispatch_owner_bitmap_range(
    std::uint64_t owner, std::uint64_t owners,
    std::uint64_t bits_per_owner, std::uint64_t* base_word,
    std::uint64_t* word_count) noexcept {
    if (base_word == nullptr || word_count == nullptr || owner >= owners)
        return false;
    if (!dispatch_bitmap_words(bits_per_owner, word_count) ||
        (*word_count != 0 &&
         owner > std::numeric_limits<std::uint64_t>::max() /
             *word_count))
        return false;
    *base_word = owner * *word_count;
    return true;
}

constexpr bool dispatch_bitmap_location(
    std::uint64_t bit, std::uint64_t bit_count,
    std::uint64_t* word, std::uint64_t* mask) noexcept {
    if (word == nullptr || mask == nullptr || bit >= bit_count)
        return false;
    *word = bit / kDispatchBitmapWordBits;
    *mask = std::uint64_t{1} << (bit % kDispatchBitmapWordBits);
    return true;
}

#if defined(DEEP_EP_ASCEND_SIMT_DEVICE)

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_bitmap_words(
    std::uint64_t bits, std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    *words = bits / kDispatchBitmapWordBits +
        (bits % kDispatchBitmapWordBits != 0 ? 1 : 0);
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_owner_bitmap_words(
    std::uint64_t owners, std::uint64_t bits_per_owner,
    std::uint64_t* words) noexcept {
    if (words == nullptr)
        return false;
    std::uint64_t words_per_owner = 0;
    if (!dispatch_simt_bitmap_words(bits_per_owner, &words_per_owner) ||
        (words_per_owner != 0 &&
         owners > std::numeric_limits<std::uint64_t>::max() /
             words_per_owner))
        return false;
    *words = owners * words_per_owner;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_owner_bitmap_range(
    std::uint64_t owner, std::uint64_t owners,
    std::uint64_t bits_per_owner, std::uint64_t* base_word,
    std::uint64_t* word_count) noexcept {
    if (base_word == nullptr || word_count == nullptr || owner >= owners)
        return false;
    if (!dispatch_simt_bitmap_words(bits_per_owner, word_count) ||
        (*word_count != 0 &&
         owner > std::numeric_limits<std::uint64_t>::max() /
             *word_count))
        return false;
    *base_word = owner * *word_count;
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ constexpr bool
dispatch_simt_bitmap_location(
    std::uint64_t bit, std::uint64_t bit_count,
    std::uint64_t* word, std::uint64_t* mask) noexcept {
    if (word == nullptr || mask == nullptr || bit >= bit_count)
        return false;
    *word = bit / kDispatchBitmapWordBits;
    *mask = std::uint64_t{1} << (bit % kDispatchBitmapWordBits);
    return true;
}

#endif

}  // namespace deep_ep::ascend::elastic
