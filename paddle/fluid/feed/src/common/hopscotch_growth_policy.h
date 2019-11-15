/**
 * MIT License
 * 
 * Copyright (c) 2018 Tessil
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef TSL_HOPSCOTCH_GROWTH_POLICY_H
#define TSL_HOPSCOTCH_GROWTH_POLICY_H 


#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <ratio>
#include <stdexcept>


namespace tsl {
namespace hh {

/**
 * Grow the hash table by a factor of GrowthFactor keeping the bucket count to a power of two. It allows
 * the table to use a mask operation instead of a modulo operation to map a hash to a bucket.
 * 
 * GrowthFactor must be a power of two >= 2.
 */
template<std::size_t GrowthFactor>
class power_of_two_growth_policy {
public:
    /**
     * Called on the hash table creation and on rehash. The number of buckets for the table is passed in parameter.
     * This number is a minimum, the policy may update this value with a higher value if needed (but not lower).
     *
     * If 0 is given, min_bucket_count_in_out must still be 0 after the policy creation and
     * bucket_for_hash must always return 0 in this case.
     */
    explicit power_of_two_growth_policy(std::size_t& min_bucket_count_in_out) {
        if(min_bucket_count_in_out > max_bucket_count()) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        if(min_bucket_count_in_out > 0) {
            min_bucket_count_in_out = round_up_to_power_of_two(min_bucket_count_in_out);
            m_mask = min_bucket_count_in_out - 1;
        }
        else {
            m_mask = 0;
        }
    }
    
    /**
     * Return the bucket [0, bucket_count()) to which the hash belongs. 
     * If bucket_count() is 0, it must always return 0.
     */
    std::size_t bucket_for_hash(std::size_t hash) const noexcept {
        return hash & m_mask;
    }
    
    /**
     * Return the bucket count to use when the bucket array grows on rehash.
     */
    std::size_t next_bucket_count() const {
        if((m_mask + 1) > max_bucket_count() / GrowthFactor) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        return (m_mask + 1) * GrowthFactor;
    }
    
    /**
     * Return the maximum number of buckets supported by the policy.
     */
    std::size_t max_bucket_count() const {
        // Largest power of two.
        return (std::numeric_limits<std::size_t>::max() / 2) + 1;
    }
    
    /**
     * Reset the growth policy as if it was created with a bucket count of 0.
     * After a clear, the policy must always return 0 when bucket_for_hash is called.
     */
    void clear() noexcept {
        m_mask = 0;
    }
    
private:
    static std::size_t round_up_to_power_of_two(std::size_t value) {
        if(is_power_of_two(value)) {
            return value;
        }
        
        if(value == 0) {
            return 1;
        }
            
        --value;
        for(std::size_t i = 1; i < sizeof(std::size_t) * CHAR_BIT; i *= 2) {
            value |= value >> i;
        }
        
        return value + 1;
    }
    
    static constexpr bool is_power_of_two(std::size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }
    
private:
    static_assert(is_power_of_two(GrowthFactor) && GrowthFactor >= 2, "GrowthFactor must be a power of two >= 2.");
    
    std::size_t m_mask;
};


/**
 * Grow the hash table by GrowthFactor::num / GrowthFactor::den and use a modulo to map a hash
 * to a bucket. Slower but it can be useful if you want a slower growth.
 */
template<class GrowthFactor = std::ratio<3, 2>>
class mod_growth_policy {
public:
    explicit mod_growth_policy(std::size_t& min_bucket_count_in_out) {
        if(min_bucket_count_in_out > max_bucket_count()) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        if(min_bucket_count_in_out > 0) {
            m_mod = min_bucket_count_in_out;
        }
        else {
            m_mod = 1;
        }
    }
    
    std::size_t bucket_for_hash(std::size_t hash) const noexcept {
        return hash % m_mod;
    }
    
    std::size_t next_bucket_count() const {
        if(m_mod == max_bucket_count()) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        const double next_bucket_count = std::ceil(double(m_mod) * REHASH_SIZE_MULTIPLICATION_FACTOR);
        if(!std::isnormal(next_bucket_count)) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        if(next_bucket_count > double(max_bucket_count())) {
            return max_bucket_count();
        }
        else {
            return std::size_t(next_bucket_count);
        }
    }
    
    std::size_t max_bucket_count() const {
        return MAX_BUCKET_COUNT;
    }
    
    void clear() noexcept {
        m_mod = 1;
    }
    
private:
    static constexpr double REHASH_SIZE_MULTIPLICATION_FACTOR = 1.0 * GrowthFactor::num / GrowthFactor::den;
    static const std::size_t MAX_BUCKET_COUNT = 
            std::size_t(double(
                    std::numeric_limits<std::size_t>::max() / REHASH_SIZE_MULTIPLICATION_FACTOR
            ));
            
    static_assert(REHASH_SIZE_MULTIPLICATION_FACTOR >= 1.1, "Growth factor should be >= 1.1.");
    
    std::size_t m_mod;
};



namespace detail {

static constexpr const std::array<std::size_t, 186> PRIMES = {{
    1ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 23ull, 29ull, 37ull, 47ull,
    59ull, 73ull, 97ull, 127ull, 151ull, 197ull, 251ull, 313ull, 397ull,
    499ull, 631ull, 797ull, 1009ull, 1259ull, 1597ull, 2011ull, 2539ull,
    3203ull, 4027ull, 5087ull, 6421ull, 8089ull, 10193ull, 12853ull, 16193ull,
    20399ull, 25717ull, 32401ull, 40823ull, 51437ull, 64811ull, 81649ull,
    102877ull, 129607ull, 163307ull, 205759ull, 259229ull, 326617ull,
    411527ull, 518509ull, 653267ull, 823117ull, 1037059ull, 1306601ull,
    1646237ull, 2074129ull, 2613229ull, 3292489ull, 4148279ull, 5226491ull,
    6584983ull, 8296553ull, 10453007ull, 13169977ull, 16593127ull, 20906033ull,
    26339969ull, 33186281ull, 41812097ull, 52679969ull, 66372617ull,
    83624237ull, 105359939ull, 132745199ull, 167248483ull, 210719881ull,
    265490441ull, 334496971ull, 421439783ull, 530980861ull, 668993977ull,
    842879579ull, 1061961721ull, 1337987929ull, 1685759167ull, 2123923447ull,
    2675975881ull, 3371518343ull, 4247846927ull, 5351951779ull, 6743036717ull,
    8495693897ull, 10703903591ull, 13486073473ull, 16991387857ull,
    21407807219ull, 26972146961ull, 33982775741ull, 42815614441ull,
    53944293929ull, 67965551447ull, 85631228929ull, 107888587883ull,
    135931102921ull, 171262457903ull, 215777175787ull, 271862205833ull,
    342524915839ull, 431554351609ull, 543724411781ull, 685049831731ull,
    863108703229ull, 1087448823553ull, 1370099663459ull, 1726217406467ull,
    2174897647073ull, 2740199326961ull, 3452434812973ull, 4349795294267ull,
    5480398654009ull, 6904869625999ull, 8699590588571ull, 10960797308051ull,
    13809739252051ull, 17399181177241ull, 21921594616111ull, 27619478504183ull,
    34798362354533ull, 43843189232363ull, 55238957008387ull, 69596724709081ull,
    87686378464759ull, 110477914016779ull, 139193449418173ull,
    175372756929481ull, 220955828033581ull, 278386898836457ull,
    350745513859007ull, 441911656067171ull, 556773797672909ull,
    701491027718027ull, 883823312134381ull, 1113547595345903ull,
    1402982055436147ull, 1767646624268779ull, 2227095190691797ull,
    2805964110872297ull, 3535293248537579ull, 4454190381383713ull,
    5611928221744609ull, 7070586497075177ull, 8908380762767489ull,
    11223856443489329ull, 14141172994150357ull, 17816761525534927ull,
    22447712886978529ull, 28282345988300791ull, 35633523051069991ull,
    44895425773957261ull, 56564691976601587ull, 71267046102139967ull,
    89790851547914507ull, 113129383953203213ull, 142534092204280003ull,
    179581703095829107ull, 226258767906406483ull, 285068184408560057ull,
    359163406191658253ull, 452517535812813007ull, 570136368817120201ull,
    718326812383316683ull, 905035071625626043ull, 1140272737634240411ull,
    1436653624766633509ull, 1810070143251252131ull, 2280545475268481167ull,
    2873307249533267101ull, 3620140286502504283ull, 4561090950536962147ull,
    5746614499066534157ull, 7240280573005008577ull, 9122181901073924329ull,
    11493228998133068689ull, 14480561146010017169ull, 18446744073709551557ull
}};

template<unsigned int IPrime>
static constexpr std::size_t mod(std::size_t hash) { return hash % PRIMES[IPrime]; }

// MOD_PRIME[iprime](hash) returns hash % PRIMES[iprime]. This table allows for faster modulo as the
// compiler can optimize the modulo code better with a constant known at the compilation.
static constexpr const std::array<std::size_t(*)(std::size_t), 186> MOD_PRIME = {{
    &mod<0>, &mod<1>, &mod<2>, &mod<3>, &mod<4>, &mod<5>, &mod<6>, &mod<7>, &mod<8>, &mod<9>, &mod<10>,
    &mod<11>, &mod<12>, &mod<13>, &mod<14>, &mod<15>, &mod<16>, &mod<17>, &mod<18>, &mod<19>, &mod<20>,
    &mod<21>, &mod<22>, &mod<23>, &mod<24>, &mod<25>, &mod<26>, &mod<27>, &mod<28>, &mod<29>, &mod<30>,
    &mod<31>, &mod<32>, &mod<33>, &mod<34>, &mod<35>, &mod<36>, &mod<37>, &mod<38>, &mod<39>, &mod<40>,
    &mod<41>, &mod<42>, &mod<43>, &mod<44>, &mod<45>, &mod<46>, &mod<47>, &mod<48>, &mod<49>, &mod<50>,
    &mod<51>, &mod<52>, &mod<53>, &mod<54>, &mod<55>, &mod<56>, &mod<57>, &mod<58>, &mod<59>, &mod<60>,
    &mod<61>, &mod<62>, &mod<63>, &mod<64>, &mod<65>, &mod<66>, &mod<67>, &mod<68>, &mod<69>, &mod<70>,
    &mod<71>, &mod<72>, &mod<73>, &mod<74>, &mod<75>, &mod<76>, &mod<77>, &mod<78>, &mod<79>, &mod<80>,
    &mod<81>, &mod<82>, &mod<83>, &mod<84>, &mod<85>, &mod<86>, &mod<87>, &mod<88>, &mod<89>, &mod<90>,
    &mod<91>, &mod<92>, &mod<93>, &mod<94>, &mod<95>, &mod<96>, &mod<97>, &mod<98>, &mod<99>, &mod<100>,
    &mod<101>, &mod<102>, &mod<103>, &mod<104>, &mod<105>, &mod<106>, &mod<107>, &mod<108>, &mod<109>, &mod<110>,
    &mod<111>, &mod<112>, &mod<113>, &mod<114>, &mod<115>, &mod<116>, &mod<117>, &mod<118>, &mod<119>, &mod<120>,
    &mod<121>, &mod<122>, &mod<123>, &mod<124>, &mod<125>, &mod<126>, &mod<127>, &mod<128>, &mod<129>, &mod<130>,
    &mod<131>, &mod<132>, &mod<133>, &mod<134>, &mod<135>, &mod<136>, &mod<137>, &mod<138>, &mod<139>, &mod<140>,
    &mod<141>, &mod<142>, &mod<143>, &mod<144>, &mod<145>, &mod<146>, &mod<147>, &mod<148>, &mod<149>, &mod<150>,
    &mod<151>, &mod<152>, &mod<153>, &mod<154>, &mod<155>, &mod<156>, &mod<157>, &mod<158>, &mod<159>, &mod<160>,
    &mod<161>, &mod<162>, &mod<163>, &mod<164>, &mod<165>, &mod<166>, &mod<167>, &mod<168>, &mod<169>, &mod<170>,
    &mod<171>, &mod<172>, &mod<173>, &mod<174>, &mod<175>, &mod<176>, &mod<177>, &mod<178>, &mod<179>, &mod<180>,
    &mod<181>, &mod<182>, &mod<183>, &mod<184>, &mod<185>
}};

}

/**
 * Grow the hash table by using prime numbers as bucket count. Slower than tsl::hh::power_of_two_growth_policy in  
 * general but will probably distribute the values around better in the buckets with a poor hash function.
 * 
 * To allow the compiler to optimize the modulo operation, a lookup table is used with constant primes numbers.
 * 
 * With a switch the code would look like:
 * \code
 * switch(iprime) { // iprime is the current prime of the hash table
 *     case 0: hash % 5ul;
 *             break;
 *     case 1: hash % 17ul;
 *             break;
 *     case 2: hash % 29ul;
 *             break;
 *     ...
 * }    
 * \endcode
 * 
 * Due to the constant variable in the modulo the compiler is able to optimize the operation
 * by a series of multiplications, substractions and shifts. 
 * 
 * The 'hash % 5' could become something like 'hash - (hash * 0xCCCCCCCD) >> 34) * 5' in a 64 bits environement.
 */
class prime_growth_policy {
public:
    explicit prime_growth_policy(std::size_t& min_bucket_count_in_out) {
        auto it_prime = std::lower_bound(detail::PRIMES.begin(), 
                                         detail::PRIMES.end(), min_bucket_count_in_out);
        if(it_prime == detail::PRIMES.end()) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        m_iprime = static_cast<unsigned int>(std::distance(detail::PRIMES.begin(), it_prime));
        if(min_bucket_count_in_out > 0) {
            min_bucket_count_in_out = *it_prime;
        }
        else {
            min_bucket_count_in_out = 0;
        }
    }
    
    std::size_t bucket_for_hash(std::size_t hash) const noexcept {
        return detail::MOD_PRIME[m_iprime](hash);
    }
    
    std::size_t next_bucket_count() const {
        if(m_iprime + 1 >= detail::PRIMES.size()) {
            throw std::length_error("The hash table exceeds its maxmimum size.");
        }
        
        return detail::PRIMES[m_iprime + 1];
    }   
    
    std::size_t max_bucket_count() const {
        return detail::PRIMES.back();
    }
    
    void clear() noexcept {
        m_iprime = 0;
    }
    
private:
    unsigned int m_iprime;
    
    static_assert(std::numeric_limits<decltype(m_iprime)>::max() >= detail::PRIMES.size(), 
                  "The type of m_iprime is not big enough.");
}; 

}
}

#endif
