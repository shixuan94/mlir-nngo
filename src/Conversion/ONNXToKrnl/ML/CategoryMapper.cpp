/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ CategoryMapper.cpp - Lowering CategoryMapper Op ---------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CategoryMapper Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using llvm::dbgs;

#define DEBUG_TYPE "category_mapper_onnx_to_krnl"

// Perform a 32 bit FNV hash on the given string
// (http://isthe.com/chongo/tech/comp/fnv).
uint32_t hash(uint32_t hval, const StringRef str) {
  constexpr uint32_t prime = 0x01000193;
  hval = (hval == 0) ? prime : hval;

  for (const char c : str) {
    hval *= prime;
    hval ^= c;
  }

  return hval;
}

uint32_t hash(uint32_t hval, const int64_t val) {
  constexpr uint32_t prime = 0x01000193;
  hval = (hval == 0) ? prime : hval;

  union {
    int64_t val;
    char arr[8];
  } u = {.val = val};

  for (int i = 0; i < 8; ++i) {
    char c = u.arr[i];
    hval *= prime;
    hval ^= c;
  }

  return hval;
}

class Utilities {
public:
  // Extracts the keys of the given map.
  template <typename KeyType, typename ValueType>
  SmallVector<KeyType> extractKeys(const std::map<KeyType, ValueType> &map) {
    SmallVector<KeyType> keys;
    for (const auto &entry : map)
      keys.push_back(entry.first);

    return keys;
  }

  // Generate the integers in the range [0 .. max-1].
  SmallVector<uint32_t> range(uint32_t max) const {
    SmallVector<uint32_t> range(max);
    std::iota(range.begin(), range.end(), 0);
    return range;
  }

  // Generate the integers in the range [min .. max-1].
  SmallVector<uint32_t> range(uint32_t min, uint32_t max) const {
    SmallVector<uint32_t> range(max - min);
    std::iota(range.begin(), range.end(), min);
    return range;
  }

  // Generate the integers [min, min+step, ...].
  SmallVector<uint32_t> range(int32_t min, int32_t max, int32_t step) const {
    SmallVector<uint32_t> range;
    int32_t nElems = (max - min) / step;
    if (nElems < 1)
      return range;

    range.resize(nElems);
    int32_t num = min;
    std::generate_n(range.begin(), nElems, [&num, step]() {
      int32_t res = num;
      num += step;
      return res;
    });
    return range;
  }

  template <typename T>
  void print(
      const SmallVectorImpl<T> &V, const Twine &name, raw_ostream &os) const {
    os << name << ": [ ";
    for (const T &elem : V)
      os << elem << ", ";
    os << "]\n\n";
  }

  template <typename T>
  void print(const SmallVectorImpl<SmallVectorImpl<T>> &V, const Twine &name,
      raw_ostream &os) const {
    os << name << ": [";
    for (const SmallVector<T> &v : V) {
      os << "[ ";
      for (const StringRef str : v)
        os << "'" << str << "' ";
      os << "] ";
    }
    os << "]\n\n";
  }

  template <typename KeyType, typename ValueType>
  void print(const std::map<KeyType, ValueType> &M, const Twine &name,
      raw_ostream &os) const {
    os << name << " : {";
    for (const auto &entry : M)
      os << "'" << entry.first << "': " << entry.second << ", ";
    os << "}\n\n";
  }
};

template <typename KeyTy, typename ValueTy>
class PerfectHash {
  // The hash table is defined by G and V.
  SmallVector<int32_t> G;
  SmallVector<int32_t> V;
  const std::map<KeyTy, ValueTy> &dict;
  Utilities utils;

public:
  PerfectHash(const std::map<KeyTy, ValueTy> &dict) : dict(dict) {
    assert(!dict.empty() && "Dictionary should not be empty");
    size_t size = dict.size();
    G.resize(size, 0);
    V.resize(size, -1);
    createPerfectHash();
  }

  const SmallVector<int32_t> &getG() const { return G; }
  const SmallVector<int32_t> &getV() const { return V; }

private:
  // Creates a minimal perfect hash for the given dictionary 'dict'.
  void createPerfectHash() {
    // Step 1: place all of the keys into buckets.
    size_t size = dict.size();
    SmallVector<KeyTy> keys = utils.extractKeys<KeyTy, ValueTy>(dict);

    SmallVector<SmallVector<KeyTy>> buckets(size);
    for (const KeyTy &key : keys)
      buckets[::hash(0, key) % size].push_back(key);

    // Step 2: Sort the buckets and process the ones with the most items first.
    llvm::sort(buckets,
        [](const SmallVectorImpl<KeyTy> &v1, const SmallVectorImpl<KeyTy> &v2) {
          return v1.size() > v2.size();
        });

    uint32_t biMax = 0;
    for (uint32_t bi : utils.range(size)) {
      biMax = bi;
      SmallVector<KeyTy> &bucket = buckets[bi];
      if (bucket.size() <= 1)
        break;

      int32_t hval = 1;
      size_t item = 0;
      SmallVector<uint32_t> slots;

      // Repeatedly try different hash values until we find a hash function that
      // places all items in the bucket into free slots.
      while (item < bucket.size()) {
        uint32_t slot = ::hash(hval, bucket[item]) % size;
        if (V[slot] != -1 ||
            std::find(slots.begin(), slots.end(), slot) != slots.end()) {
          hval++;
          item = 0;
          slots.clear();
        } else {
          slots.push_back(slot);
          item++;
        }
      }

      G[::hash(0, bucket[0]) % size] = hval;
      for (uint32_t i : utils.range(bucket.size()))
        V[slots[i]] = dict.at(bucket[i]);
    }

    // Place remaining buckets (containing a single entry) into a free slot. Use
    // a negative value of hval to indicate this.
    SmallVector<uint32_t> freeList;
    for (uint32_t i : utils.range(size))
      if (V[i] == -1)
        freeList.push_back(i);

    for (uint32_t i : utils.range(biMax, size)) {
      SmallVector<KeyTy> &bucket = buckets[i];
      if (bucket.size() == 0)
        break;

      uint32_t slot = freeList.back();
      freeList.pop_back();

      // Subtract one to ensure it's negative even if the zeroeth slot was used.
      G[::hash(0, bucket[0]) % size] = -(int32_t)slot - 1;
      V[slot] = dict.at(bucket[0]);
    }
  }
};

// TODO: this will be placed in the runtime library.
// Return the index corresponding to the given word using the perfect hash
// table defined by G and V. The perfect hash table algorithm requires the
// word to exists in the dictionary, if it doesn't this function will return a
// valid index anyway. The caller should confirm that the word at that index
// is equal to the one looked up.
template <typename KeyTy>
uint32_t getIndex(
    const KeyTy *key, const int32_t G[], const int32_t V[], int32_t len) {
  int32_t d = G[::hash(0, key) % len];
  int32_t index = (d < 0) ? V[-d - 1] : V[::hash(d, key) % len];
  assert(index >= 0 && index < len && "Out of bounds index");
  return index;
}

#include <fstream>

template <typename KeyTy, typename ValueTy>
void createDictionary(
    std::map<KeyTy, ValueTy> &dict, std::vector<KeyTy> &words) {
  std::ifstream dictionary("/Users/etiotto@ca.ibm.com/tmp/words1");

  int64_t lineNum = 0;
  std::string word;
  while (getline(dictionary, word)) {
    words.push_back(word);
    dict[word] = lineNum++;
  }

  LLVM_DEBUG({
    Utilities utils;
    utils.print(dict, "dict", dbgs());
  });
}

void test() {
  // Read in the dictionary (assume one word per line).
  using KeyTy = std::string;
  using ValueTy = int64_t;

  std::map<KeyTy, ValueTy> dict;
  std::vector<KeyTy> words;
  createDictionary<KeyTy, ValueTy>(dict, words);

  // Construct the perfect hash for the dictionary (at compile time).
  PerfectHash<KeyTy, ValueTy> perfectHash(dict);

  // Lookup words using the runtime.
  static const char *testWords[] = {
      "hello", "goodbye", "dog", "cat", "ettore", "alex", "whitney"};

  const SmallVector<int32_t> G = perfectHash.getG();
  const SmallVector<int32_t> V = perfectHash.getV();

  for (const char *word : testWords) {
    int64_t index = getIndex(word, &G[0], &V[0], dict.size());
    LLVM_DEBUG(dbgs() << "index = " << index << ", word: " << word << "\n";);
    // confirm word exists in the dictionary
    if (strncmp(word, words[index].c_str(), strlen(word)) != 0)
      LLVM_DEBUG(dbgs() << word << " not in dictionary\n";);
  }
}

struct ONNXCategoryMapperOpLowering : public ConversionPattern {
  ONNXCategoryMapperOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXCategoryMapperOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto categoryMapperOp = cast<ONNXCategoryMapperOp>(op);
    ONNXCategoryMapperOpAdaptor operandAdaptor(operands);

    ONNXCategoryMapperOpShapeHelper shapeHelper(&categoryMapperOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    // Operands and attributes.
    Location loc = categoryMapperOp.getLoc();
    Value X = operandAdaptor.X();
    ArrayAttr cats_int64s = categoryMapperOp.cats_int64sAttr();
    ArrayAttr cats_strings = categoryMapperOp.cats_stringsAttr();
    IntegerAttr default_int64 = categoryMapperOp.default_int64Attr();
    StringAttr default_string = categoryMapperOp.default_stringAttr();

    // Basic information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();
    ShapedType inputType = X.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    OnnxBuilder createOnnx(rewriter, op->getLoc());
    KrnlBuilder createKrnl(createOnnx);
    MathBuilder createMath(createKrnl);

    Value constantForG, constantForV;

    // Populate the dictionary.
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // Populate The dictionary.
          assert(type.getWidth() == 64 && type.isSignedInteger() &&
                 "Unexpected integer type");
          std::map<int64_t, int32_t> dict;
          std::vector<int64_t> keys;
          int32_t size = cats_int64s.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_int64s, idx);
            int64_t key = elemAttr.cast<IntegerAttr>().getSInt();
            dict[key] = idx;
            keys.push_back(key);
          }

          // Create The perfect hash.
          PerfectHash<int64_t, int32_t> perfectHash(dict);

          constantForG = createOnnx.constant(
              rewriter.getI32TensorAttr(perfectHash.getG()));
          constantForV = createOnnx.constant(
              rewriter.getI32TensorAttr(perfectHash.getV()));
        })
        .Case<onnxmlir::StringType>([&](onnxmlir::StringType type) {
          // Populate The dictionary.
          std::map<StringRef, int32_t> dict;
          std::vector<StringRef> keys;
          int32_t size = cats_strings.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_strings, idx);
            StringRef key = elemAttr.cast<StringAttr>().getValue();
            dict[key] = idx;
            keys.push_back(key);
          }

          // Create The perfect hash.
          PerfectHash<StringRef, int32_t> perfectHash(dict);

          constantForG = createOnnx.constant(
              rewriter.getI32TensorAttr(perfectHash.getG()));
          constantForV = createOnnx.constant(
              rewriter.getI32TensorAttr(perfectHash.getV()));
        })
        .Default([&](Type type) { llvm_unreachable("Illegal KeyTy"); });

    // Create loop invariant values.
    Value constantForCatsInt64s = createKrnl.constant(
        MemRefType::get({static_cast<int64_t>(cats_int64s.size())},
            rewriter.getIntegerType(64)),
        "cat_int64s", cats_int64s);
    Value constantForCatsStrings = createKrnl.constant(
        MemRefType::get({static_cast<int64_t>(cats_strings.size())},
            onnxmlir::StringType::get(rewriter.getContext())),
        "cat_strings", cats_strings);

    Value dictSize =
        createMath.constant(rewriter.getI32Type(), cats_strings.size());
    Value zeroVal = createMath.constant(rewriter.getIntegerType(32), 0);
    Value defaultInt64Val = createMath.constant(
        rewriter.getIntegerType(64), default_int64.getSInt());

    // Lookup the Key index for each input value.
    BuildKrnlLoop inputLoops(rewriter, loc, rank);
    inputLoops.createDefineAndIterateOp(X);
    rewriter.setInsertionPointToStart(inputLoops.getIterateBlock());
    {
      // Get a child IndexExpr context.
      // IndexExprScope childScope(&rewriter, shapeHelper.scope);

      // Get indices.
      SmallVector<IndexExpr, 4> IVs;
      for (decltype(rank) i = 0; i < rank; ++i) {
        Value iv = inputLoops.getInductionVar(i);
        IVs.emplace_back(DimIndexExpr(iv));
      }

      // Generate the 'findIndex' call.
      Value inputElem = createKrnl.loadIE(X, IVs);
      Value index = rewriter.create<KrnlFindIndexOp>(
          loc, rewriter.getI32Type(), inputElem, constantForG, constantForV);

      // Generate the 'strncmp' call.
      Value strlen; // TODO generate strlen(word);
      Value str = createKrnl.load(constantForCatsStrings, {index});
      Value strncmpRes = rewriter.create<KrnlStrncmpOp>(
          loc, rewriter.getI32Type(), inputElem, str, dictSize);
      Value isIndexValid = createMath.eq(strncmpRes, zeroVal);

      // generate an if statement to check whether the index is valid.
      scf::IfOp ifOp = rewriter.create<scf::IfOp>(
          loc, isIndexValid, /*withElseRegion=*/true);

      // If the index is valid, retrieve the value at 'index' from 'cat_int64s'
      // array and store it into the result buffer.
      {
        rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
        Value loadData = createKrnl.load(constantForCatsInt64s, {index});
        createKrnl.storeIE(loadData, alloc, IVs);
      }

      // If the index is not valid, store the default value into the result
      // buffer.
      {
        rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
        createKrnl.storeIE(defaultInt64Val, alloc, IVs);
      }
    }

    rewriter.replaceOp(op, alloc);

    LLVM_DEBUG({
      FuncOp function = getContainingFunction(op);
      assert(function && "Could not find parent function");
      dbgs() << "function: " << function << "\n";
    });

    return success();
  }

private:
  static Attribute getElemAttr(ArrayAttr arr, int32_t idx) {
    return arr.getValue()[idx];
  }

#if 0
  DenseElementsAttr createDenseArrayAttr(
      ArrayAttr origAttr, Type elemType) const {
    DenseElementsAttr Result;
    TypeSwitch<Type>(elemType)
        .Case<IntegerType>([&](IntegerType type) {
          SmallVector<int64_t, 4> tmp;
          for (unsigned i = 0; i < origAttr.size(); ++i)
            tmp.emplace_back(
                origAttr.getValue()[i].dyn_cast<IntegerAttr>().getInt());

          Result = DenseElementsAttr::get(
              RankedTensorType::get(tmp.size(), elemType),
              llvm::makeArrayRef(tmp));
        })
        .Case<onnxmlir::StringType>([&](onnxmlir::StringType type) {
          SmallVector<StringRef, 4> tmp;
          for (unsigned i = 0; i < origAttr.size(); ++i)
            tmp.emplace_back(
                origAttr.getValue()[i].dyn_cast<StringAttr>().getValue());

          Result = DenseElementsAttr::get(
              RankedTensorType::get(tmp.size(), elemType),
              llvm::makeArrayRef(tmp));
        })
        .Default([&](Type type) { llvm_unreachable("Illegal KeyTy"); });

    return Result;
  }

  static Optional<FlatSymbolRefAttr> getFunctionDeclaration(
      ModuleOp module, const char *funcName) {
    assert(funcName && "Missing function name");
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(module.getContext(), funcName);
    return None;
  }

  /// Return a symbol reference to the strncmp function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertStrncmpDecl(
      PatternRewriter &rewriter, ModuleOp module) {
    constexpr const char *funcName = "strncmp";
    Optional<FlatSymbolRefAttr> optFuncDecl =
        getFunctionDeclaration(module, funcName);
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'strncmp' function signature: `i32 (i8*, i8*, i64)`
    auto *ctx = module.getContext();
    auto i8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto fnType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(),
        ArrayRef<Type>({i8PtrTy, i8PtrTy, rewriter.getI64Type()}), false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto oldInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());
    // rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);
    rewriter.restoreInsertionPoint(oldInsertionPoint);

    return SymbolRefAttr::get(ctx, funcName);
  }

  /// Return a symbol reference to the runtime findIndex function, inserting
  /// it into the module if necessary.
  static FlatSymbolRefAttr getOrInsertFindIndexDecl(
      PatternRewriter &rewriter, ModuleOp module) {
    // TODO: use getOrInsertExternFunc
    constexpr const char *funcName = "findIndex";
    Optional<FlatSymbolRefAttr> optFuncDecl =
        getFunctionDeclaration(module, funcName);
    if (optFuncDecl.hasValue())
      return optFuncDecl.getValue();

    // Create 'findIndex' function signature: `i32 (i8*, i32*, i32*, i64)`
    auto *ctx = module.getContext();
    auto i8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto i32PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 32));
    auto fnType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(),
        ArrayRef<Type>({i8PtrTy, i32PtrTy, i32PtrTy, rewriter.getI64Type()}),
        false);

    // Insert the function declaration the module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto oldInsertionPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fnType);
    rewriter.restoreInsertionPoint(oldInsertionPoint);

    return SymbolRefAttr::get(ctx, funcName);
  }
#endif
};

void populateLoweringONNXCategoryMapperOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCategoryMapperOpLowering>(ctx);
}
