/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- OMSignature.h - OMSignature Declaration header -----------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMSignature API functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#pragma once

extern "C" {
#endif

/**
 * \brief Return the input signature of the given entry point as a JSON string.
 *
 * @param entry point name
 * @return pointer to input signature JSON string
 */
const char *omInputSignature(const char *entryPointName);

/**
 * \brief Return the output signature of the given entry point as a JSON string.
 *
 * @param entry point name
 * @return pointer to output signature JSON string
 */
const char *omOutputSignature(const char *entryPointName);

#ifdef __cplusplus
}
#endif
