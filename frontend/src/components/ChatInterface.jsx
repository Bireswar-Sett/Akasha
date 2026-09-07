import React, { useState, useRef, useEffect } from 'react';

import {
  Send,
  Paperclip,
  X,
  Bot,
  User,
  Image as ImageIcon,
  ArrowUpRight
} from 'lucide-react';

import axios from 'axios';

import logoSrc from '../assets/logo.png';

import {
  auth,
  storage,
  db,
  isDemoMode
} from '../firebaseClient';

import {
  ref,
  uploadBytes
} from 'firebase/storage';

import {
  collection,
  addDoc
} from 'firebase/firestore';


/* ============================================================================
 * Upload limits / supported source formats
 * ========================================================================== */

const MAX_UPLOAD_FILES = 4;

const SUPPORTED_EXTENSIONS = new Set([
  '.tif',
  '.tiff',
  '.png',
  '.jpg',
  '.jpeg'
]);


/* ============================================================================
 * Derived-artifact protection
 * ========================================================================== */

const DERIVED_ARTIFACT_PATTERNS = [
  /(^|[\-_])pseudo[\-_]?rgb([\-_]|$)/i,
  /(^|[\-_])preview([\-_]|$)/i,
  /(^|[\-_])thumbnail([\-_]|$)/i,
  /(^|[\-_])derived([\-_]|$)/i
];


const isDerivedArtifact = (filename = '') => {
  return DERIVED_ARTIFACT_PATTERNS.some(
    pattern => pattern.test(filename)
  );
};


/* ============================================================================
 * File helpers
 * ========================================================================== */

const getExtension = (filename = '') => {
  const match = filename
    .toLowerCase()
    .match(/(\.[a-z0-9]+)$/);

  return match ? match[1] : '';
};


const getBaseName = (filename = '') => {
  const extension = getExtension(filename);

  return extension
    ? filename.slice(0, -extension.length)
    : filename;
};


const sanitizeIdPart = (value = '') => {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
};


/* ============================================================================
 * Remote-sensing filename hints
 *
 * These are CLIENT-SIDE HINTS only.
 *
 * The backend remains authoritative and must validate trusted metadata.
 * ========================================================================== */

const inferModality = (filename = '') => {
  const lower = filename.toLowerCase();

  const looksSar =
    lower.includes('sentinel-1') ||
    lower.includes('sentinel_1') ||
    lower.includes('sar') ||
    lower.includes('gamma0') ||
    lower.includes('gamma_0') ||
    lower.includes('sigma0') ||
    lower.includes('sigma_0') ||
    /(^|[\-_])(vv|vh)(?=[_.-]|$)/i.test(lower);

  if (looksSar) {
    return 'sar';
  }

  const looksOptical =
    lower.includes('sentinel-2') ||
    lower.includes('sentinel_2') ||
    lower.includes('true_color') ||
    lower.includes('truecolor') ||
    lower.includes('optical') ||
    lower.includes('multispectral') ||
    lower.includes('landsat') ||
    lower.includes('cartosat') ||
    lower.includes('planet');

  if (looksOptical) {
    return 'optical';
  }

  /*
   * Standard consumer image formats can safely default to optical
   * because they are already intended as visual imagery.
   *
   * Unknown TIFF/GeoTIFF must NOT silently become optical.
   */
  if (
    getExtension(filename) === '.png' ||
    getExtension(filename) === '.jpg' ||
    getExtension(filename) === '.jpeg'
  ) {
    return 'optical';
  }

  return null;
};


const inferPolarization = (filename = '') => {
  const lower = filename.toLowerCase();

  const matches = [
    ...lower.matchAll(
      /(?:^|[\-_])(vv|vh)(?=[_.-]|$)/gi
    )
  ];

  if (matches.length === 0) {
    return null;
  }

  /*
   * Sentinel-1 filenames can contain strings such as:

       ...VV_VH_VH_decibel_gamma0.tiff
       ...VV_VH_VV_decibel_gamma0.tiff

   * The final standalone VV/VH token is the actual channel identifier
   * for the source file.
   */
  return matches[matches.length - 1][1].toUpperCase();
};


const inferAcquisitionTime = (filename = '') => {
  /*
   * Handles names such as:

       2024-01-04-00_00_2024-01-04-23_59_...
       2026-08-25-00_00_2026-08-25-23_59_...

   * The first YYYY-MM-DD occurrence is used.
   */
  const match = filename.match(
    /(20\d{2})[-_](\d{2})[-_](\d{2})/
  );

  if (!match) {
    return null;
  }

  const [, year, month, day] = match;

  return `${year}-${month}-${day}`;
};


/* ============================================================================
 * SAR acquisition grouping
 * ========================================================================== */

/*
 * Remove ALL polarization tokens from a SAR filename so VV and VH files
 * belonging to one acquisition produce the same grouping seed.
 *
 * Example:
 *
 *   foo_VV_VH_VV_decibel_gamma0
 *   foo_VV_VH_VH_decibel_gamma0
 *
 * both become approximately:
 *
 *   foo_decibel_gamma0
 */
const removePolarizationTokens = (filename = '') => {
  return filename
    .replace(
      /(?:^|[\-_])(vv|vh)(?=[_.-]|$)/gi,
      '_'
    )
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
};


const buildSarObservationSeed = (
  filename,
  acquisitionTime
) => {
  const baseName = getBaseName(filename);

  const sourcePart = sanitizeIdPart(
    removePolarizationTokens(baseName)
  );

  return [
    acquisitionTime || 'unknown_date',
    sourcePart || 'unknown_source'
  ].join('_');
};


/* ============================================================================
 * Upload descriptors
 * ========================================================================== */

const buildUploadDescriptors = files => {
  const sarGroups = new Map();

  const descriptors = files.map((file, index) => {
    const filename =
      file?.name ||
      `image_${index + 1}`;

    const extension = getExtension(filename);

    if (!SUPPORTED_EXTENSIONS.has(extension)) {
      throw new Error(
        `Unsupported imagery format: ${filename}`
      );
    }

    if (isDerivedArtifact(filename)) {
      throw new Error(
        `"${filename}" appears to be a derived preview/artifact, ` +
        `not source satellite imagery. Please upload the original ` +
        `optical image or original SAR VV/VH files.`
      );
    }

    const modality = inferModality(filename);

    if (!modality) {
      throw new Error(
        `Could not determine whether "${filename}" is optical or SAR. ` +
        `Please use a recognizable remote-sensing filename.`
      );
    }

    const acquisitionTime =
      inferAcquisitionTime(filename);

    const polarization =
      modality === 'sar'
        ? inferPolarization(filename)
        : null;

    if (
      modality === 'sar' &&
      !polarization
    ) {
      throw new Error(
        `Could not determine VV/VH polarization from "${filename}". ` +
        `A SAR observation requires both VV and VH files.`
      );
    }

    let observationId;

    if (modality === 'sar') {
      const sarSeed =
        buildSarObservationSeed(
          filename,
          acquisitionTime
        );

      if (!sarGroups.has(sarSeed)) {
        sarGroups.set(
          sarSeed,
          `obs_sar_${sanitizeIdPart(sarSeed)}`
        );
      }

      observationId =
        sarGroups.get(sarSeed);
    } else {
      observationId =
        `obs_optical_${sanitizeIdPart(
          acquisitionTime || `unknown_${index + 1}`
        )}_${index + 1}`;
    }

    return {
      file,
      filename,
      extension,
      modality,
      polarization,
      acquisitionTime,
      observationId
    };
  });

  return descriptors;
};


/* ============================================================================
 * Logical upload validation
 * ========================================================================== */

const validateUploadCombination = descriptors => {
  if (!descriptors.length) {
    throw new Error(
      'At least one satellite image is required.'
    );
  }

  if (descriptors.length > MAX_UPLOAD_FILES) {
    throw new Error(
      `You can upload at most ${MAX_UPLOAD_FILES} physical files.`
    );
  }

  const sarDescriptors =
    descriptors.filter(
      item => item.modality === 'sar'
    );

  const opticalDescriptors =
    descriptors.filter(
      item =>
        item.modality === 'optical'
    );

  /* ------------------------------------------------------------------------
   * SAR logical grouping
   * ---------------------------------------------------------------------- */

  const sarGroups = new Map();

  for (const item of sarDescriptors) {
    if (!sarGroups.has(item.observationId)) {
      sarGroups.set(
        item.observationId,
        []
      );
    }

    sarGroups
      .get(item.observationId)
      .push(item);
  }

  for (
    const [observationId, group]
    of sarGroups.entries()
  ) {
    const vvCount =
      group.filter(
        item => item.polarization === 'VV'
      ).length;

    const vhCount =
      group.filter(
        item => item.polarization === 'VH'
      ).length;

    if (
      vvCount !== 1 ||
      vhCount !== 1
    ) {
      throw new Error(
        `SAR observation "${observationId}" must contain exactly ` +
        `one VV file and one VH file.`
      );
    }
  }

  /* ------------------------------------------------------------------------
   * Valid physical configurations
   *
   * 1  optical
   * 2  optical
   * 2  SAR
   * 3  optical + SAR(VV,VH)
   * 4  SAR(T1) + SAR(T2)
   * ---------------------------------------------------------------------- */

  const valid =
    descriptors.length === 1 ||
    (
      descriptors.length === 2 &&
      (
        sarGroups.size === 1 ||
        opticalDescriptors.length === 2
      )
    ) ||
    (
      descriptors.length === 3 &&
      opticalDescriptors.length === 1 &&
      sarGroups.size === 1
    ) ||
    (
      descriptors.length === 4 &&
      sarGroups.size === 2 &&
      opticalDescriptors.length === 0
    );

  if (!valid) {
    throw new Error(
      'The selected imagery cannot be represented as supported logical ' +
      'observations. Supported combinations are: single optical, single SAR ' +
      '(VV+VH), two optical observations, optical+SAR, or two SAR observations.'
    );
  }

  /*
   * A single SAR observation = exactly two physical files.
   */
  for (
    const [, group]
    of sarGroups.entries()
  ) {
    if (group.length !== 2) {
      throw new Error(
        'Each SAR acquisition must contain exactly one VV and one VH file.'
      );
    }
  }

  return {
    sarCount: sarGroups.size,
    opticalCount: opticalDescriptors.length,
    logicalObservationCount:
      sarGroups.size +
      opticalDescriptors.length
  };
};


/* ============================================================================
 * Relationship inference
 *
 * This is only a request-side grouping hint.
 * The backend remains authoritative.
 * ========================================================================== */

const buildRelationshipHint = descriptors => {
  const sar = descriptors.filter(
    item => item.modality === 'sar'
  );

  const optical = descriptors.filter(
    item => item.modality === 'optical'
  );

  const sarObservationIds = [
    ...new Set(
      sar.map(item => item.observationId)
    )
  ];

  /* Single optical */
  if (
    descriptors.length === 1 &&
    optical.length === 1
  ) {
    return {
      type: 'single'
    };
  }

  /* Single SAR acquisition */
  if (
    descriptors.length === 2 &&
    sarObservationIds.length === 1
  ) {
    return {
      type: 'single'
    };
  }

  /* Two optical observations */
  if (
    descriptors.length === 2 &&
    optical.length === 2
  ) {
    const timestamps =
      optical
        .map(item => item.acquisitionTime)
        .filter(Boolean);

    return {
      type:
        timestamps.length === 2
          ? 'bi_temporal'
          : 'temporal'
    };
  }

  /* Optical + SAR */
  if (
    descriptors.length === 3 &&
    optical.length === 1 &&
    sarObservationIds.length === 1
  ) {
    return {
      type: 'cross_modal'
    };
  }

  /* Two SAR observations */
  if (
    descriptors.length === 4 &&
    sarObservationIds.length === 2
  ) {
    return {
      type: 'bi_temporal'
    };
  }

  return {
    type: 'single'
  };
};


/* ============================================================================
 * Build backend-facing manifest hint
 * ========================================================================== */

const buildClientManifest = descriptors => {
  const physicalFiles =
    descriptors.map(
      (item, index) => ({
        id: `file_${index}`,
        filename: item.filename,
        format: item.extension
          .replace('.', '')
          .toLowerCase(),
        modality: item.modality,
        polarization:
          item.polarization || null,
        timestamp:
          item.acquisitionTime || null,
        observation_id:
          item.observationId,
        role:
          item.modality === 'sar'
            ? 'source_sar_channel'
            : 'source_image'
      })
    );

  const observationMap = new Map();

  for (
    let index = 0;
    index < descriptors.length;
    index += 1
  ) {
    const descriptor =
      descriptors[index];

    if (
      !observationMap.has(
        descriptor.observationId
      )
    ) {
      observationMap.set(
        descriptor.observationId,
        {
          id: descriptor.observationId,
          modality: descriptor.modality,
          acquisition_time:
            descriptor.acquisitionTime ||
            null,
          metadata: {}
        }
      );
    }

    const observation =
      observationMap.get(
        descriptor.observationId
      );

    if (descriptor.modality === 'sar') {
      if (!observation.sar) {
        observation.sar = {};
      }

      observation.sar[
        descriptor.polarization.toLowerCase()
      ] = {
        physical_index: index,
        id: `file_${index}`
      };
    } else {
      observation.image = {
        physical_index: index,
        id: `file_${index}`
      };
    }
  }

  return {
    physical_files,
    observations: [
      ...observationMap.values()
    ],
    relationship:
      buildRelationshipHint(
        descriptors
      ),
    metadata: {
      source: 'akasha_frontend',
      version: '3'
    }
  };
};


/* ============================================================================
 * Component
 * ========================================================================== */

const ChatInterface = ({
  selectedFiles,
  onFileSelect,
  activeSession,
  onUpdateSessionMessages,
  user,
  draftQuery,
  onDraftQueryChange,
  onClearDraft
}) => {
  const query =
    draftQuery || '';

  const setQuery =
    onDraftQueryChange;

  const [
    isProcessing,
    setIsProcessing
  ] = useState(false);

  const [
    previewImage,
    setPreviewImage
  ] = useState(null);

  const messagesEndRef =
    useRef(null);


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth'
    });
  };


  useEffect(() => {
    scrollToBottom();
  }, [
    activeSession?.messages,
    isProcessing
  ]);


  /* ========================================================================
   * Send
   * ====================================================================== */

  const handleSend = async (
    overrideQuery = null
  ) => {
    let textToSend =
      overrideQuery || query;

    if (
      (!textToSend ||
        !textToSend.trim()) &&
      selectedFiles &&
      selectedFiles.length > 0
    ) {
      textToSend =
        'Analyze attached satellite imagery.';
    }

    if (
      !textToSend ||
      !textToSend.trim() ||
      isProcessing
    ) {
      return;
    }

    const filesToUpload = (
      selectedFiles
        ? [...selectedFiles]
        : []
    ).slice(
      0,
      MAX_UPLOAD_FILES
    );

    /* ----------------------------------------------------------------------
     * Build logical descriptors before touching Storage.
     * -------------------------------------------------------------------- */

    let descriptors = [];

    try {
      descriptors =
        buildUploadDescriptors(
          filesToUpload
        );

      validateUploadCombination(
        descriptors
      );
    } catch (validationError) {
      const botError = {
        id: (
          Date.now() + 1
        ).toString(),

        sender: 'assistant',

        isError: true,

        text:
          validationError?.message ||
          'Unsupported imagery combination.',

        timestamp:
          new Date().toLocaleTimeString(
            [],
            {
              hour: '2-digit',
              minute: '2-digit'
            }
          )
      };

      onUpdateSessionMessages([
        ...(activeSession?.messages || []),

        {
          id:
            Date.now().toString(),

          sender: 'user',

          text: textToSend,

          attachments:
            filesToUpload.map(
              file => ({
                name: file.name,
                url:
                  URL.createObjectURL(
                    file
                  )
              })
            ),

          timestamp:
            new Date().toLocaleTimeString(
              [],
              {
                hour: '2-digit',
                minute: '2-digit'
              }
            )
        },

        botError
      ]);

      return;
    }

    if (onClearDraft) {
      onClearDraft();
    }

    setIsProcessing(true);


    /* ========================================================================
     * User message
     * ====================================================================== */

    const userMessage = {
      id:
        Date.now().toString(),

      sender: 'user',

      text: textToSend,

      attachments: [],

      timestamp:
        new Date().toLocaleTimeString(
          [],
          {
            hour: '2-digit',
            minute: '2-digit'
          }
        )
    };


    const messagesWithUser = [
      ...(activeSession?.messages || []),
      userMessage
    ];

    onUpdateSessionMessages(
      messagesWithUser
    );


    /* ========================================================================
     * Upload imagery
     * ====================================================================== */

    const attachmentUrls = [];

    const uploadedImagePaths = [];

    const uploadedImageMetadata = [];

    try {
      for (
        let index = 0;
        index < descriptors.length;
        index += 1
      ) {
        const descriptor =
          descriptors[index];

        const {
          file,
          filename,
          extension,
          modality,
          polarization,
          acquisitionTime,
          observationId
        } = descriptor;

        let localPreviewUrl = '';

        let storagePath = '';

        if (
          user &&
          !isDemoMode
        ) {
          const uid =
            user.uid ||
            user.id ||
            user.firebaseUser?.uid;

          if (!uid) {
            throw new Error(
              'Logged-in user does not have a valid Firebase UID.'
            );
          }

          const timestamp =
            Date.now();

          storagePath =
            `users/${uid}/imagery/` +
            `${timestamp}_${file.name}`;

          const storageRef =
            ref(
              storage,
              storagePath
            );


          /*
           * Firebase custom metadata.
           *
           * These are intended to let the backend reconstruct the logical
           * observations from the physical files.
           *
           * The backend MUST still validate these values and should never
           * trust arbitrary request-body metadata over Storage metadata.
           */
          const customMetadata = {
            original_filename:
              filename,

            file_format:
              extension
                .replace('.', '')
                .toLowerCase(),

            modality,

            acquisition_time:
              acquisitionTime || '',

            observation_id:
              observationId,

            polarization:
              polarization || '',

            imagery_role:
              modality === 'sar'
                ? 'source_sar_channel'
                : 'source_image',

            uploaded_by:
              'akasha_frontend',

            metadata_version:
              '3'
          };


          await uploadBytes(
            storageRef,
            file,
            {
              customMetadata
            }
          );


          /*
           * Firestore metadata is useful for displaying the user's imagery
           * library, but is NOT the security authority for analysis.
           */
          await addDoc(
            collection(
              db,
              'users',
              uid,
              'imagery'
            ),
            {
              name: filename,

              path: storagePath,

              uploadedAt:
                Date.now(),

              size: file.size,

              modality,

              polarization:
                polarization || null,

              acquisitionTime:
                acquisitionTime || null,

              observationId,

              format:
                extension
                  .replace('.', '')
                  .toLowerCase(),

              metadataVersion:
                '3'
            }
          );


          uploadedImagePaths.push(
            storagePath
          );


          uploadedImageMetadata.push({
            path: storagePath,

            filename,

            format:
              extension
                .replace('.', '')
                .toLowerCase(),

            modality,

            polarization:
              polarization || null,

            acquisition_time:
              acquisitionTime || null,

            observation_id:
              observationId,

            physical_index:
              index
          });


          localPreviewUrl =
            URL.createObjectURL(
              file
            );

        } else {
          /*
           * Demo/local mode.
           */
          localPreviewUrl =
            URL.createObjectURL(
              file
            );

          uploadedImageMetadata.push({
            path: null,

            filename,

            format:
              extension
                .replace('.', '')
                .toLowerCase(),

            modality,

            polarization:
              polarization || null,

            acquisition_time:
              acquisitionTime || null,

            observation_id:
              observationId,

            physical_index:
              index
          });
        }


        attachmentUrls.push({
          name: filename,

          url: localPreviewUrl,

          modality,

          polarization:
            polarization || null,

          acquisitionTime:
            acquisitionTime || null,

          observationId
        });
      }

    } catch (uploadErr) {
      console.error(
        'Storage upload error:',
        uploadErr
      );

      const uploadErrorMessage =
        uploadErr?.message?.includes(
          'permission'
        )
          ? 'Storage permission denied. Please check your Firebase Storage rules.'
          : (
            uploadErr?.message ||
            'Image upload failed. Please try again.'
          );

      const userMessageWithAttachments = {
        ...userMessage,

        attachments:
          attachmentUrls
      };

      const updatedMessages =
        messagesWithUser.map(
          message =>
            message.id ===
            userMessage.id
              ? userMessageWithAttachments
              : message
        );

      onUpdateSessionMessages([
        ...updatedMessages,

        {
          id:
            (
              Date.now() + 1
            ).toString(),

          sender: 'assistant',

          isError: true,

          text:
            uploadErrorMessage,

          timestamp:
            new Date().toLocaleTimeString(
              [],
              {
                hour: '2-digit',
                minute: '2-digit'
              }
            )
        }
      ]);

      setIsProcessing(false);

      return;
    }


    const userMessageWithAttachments = {
      ...userMessage,

      attachments:
        attachmentUrls
    };

    const updatedMessages =
      messagesWithUser.map(
        message =>
          message.id ===
          userMessage.id
            ? userMessageWithAttachments
            : message
      );

    onUpdateSessionMessages(
      updatedMessages
    );


    /* ========================================================================
     * Backend analysis request
     * ====================================================================== */

    try {
      if (
        !user &&
        !isDemoMode
      ) {
        throw new Error(
          'You must be signed in to analyze imagery.'
        );
      }

      const currentFirebaseUser =
        auth?.currentUser;

      const idToken =
        currentFirebaseUser
          ? await currentFirebaseUser.getIdToken()
          : isDemoMode
            ? 'demo-local-token'
            : null;

      if (!idToken) {
        throw new Error(
          'Authentication required before analysis can run.'
        );
      }


      /*
       * Build the complete logical manifest.
       *
       * Important:
       * image_paths remain the physical transport layer.
       * client_manifest describes logical observations.
       */
      const clientManifest =
        buildClientManifest(
          descriptors
        );


      const requestBody = {
        query: textToSend,

        /*
         * Backward-compatible first physical file.
         */
        image_path:
          uploadedImagePaths[0] ||
          null,

        /*
         * Physical transport layer.
         */
        image_paths:
          uploadedImagePaths,

        /*
         * Normalized physical metadata hints.
         */
        image_metadata:
          uploadedImageMetadata,

        /*
         * Explicit logical observation structure.
         */
        input_manifest:
          clientManifest,

        max_new_tokens:
          256
      };


      const response =
        await axios.post(
          '/api/analyze',
          requestBody,
          {
            // Qwen orchestration can legitimately exceed a short browser
            // timeout while the API remains healthy.
            timeout: 180000,

            headers: {
              Authorization:
                `Bearer ${idToken}`
            }
          }
        );


      const data =
        response.data;


      const botMessage = {
        id:
          (
            Date.now() + 1
          ).toString(),

        sender: 'assistant',

        text:
          data.answer ||
          'Analysis completed without a textual response.',

        timestamp:
          new Date().toLocaleTimeString(
            [],
            {
              hour: '2-digit',
              minute: '2-digit'
            }
          )
      };


      onUpdateSessionMessages([
        ...updatedMessages,
        botMessage
      ]);

    } catch (error) {
      console.error(
        'Backend error:',
        error?.response?.status,
        error?.message
      );

      const status =
        error?.response?.status;

      const detail =
        error?.response?.data?.detail ||
        '';

      const isTimeout =
        error?.code === 'ECONNABORTED';

      const isOffline =
        !error?.response &&
        !isTimeout;

      let errorMessage;


      if (
        detail.includes(
          'ZeroGPU'
        ) ||
        detail.includes(
          'runs limit'
        ) ||
        detail.includes(
          'daily AI analysis limit'
        )
      ) {
        errorMessage =
          'The AI specialist quota is currently unavailable. Please try again later.';

      } else if (
        status === 400 &&
        detail
      ) {
        errorMessage =
          `Analysis failed (400): ${detail}`;

      } else if (isTimeout) {
        errorMessage =
          'Analysis is taking longer than expected. The API is reachable; please try again shortly.';

      } else if (isOffline) {
        errorMessage =
          'Backend offline. Ensure the API service is running.';

      } else {
        errorMessage =
          `Analysis failed (${status || 'error'}): ${
            detail ||
            error?.message ||
            'Please try again.'
          }`;
      }


      const botError = {
        id:
          (
            Date.now() + 1
          ).toString(),

        sender: 'assistant',

        isError: true,

        text:
          errorMessage,

        timestamp:
          new Date().toLocaleTimeString(
            [],
            {
              hour: '2-digit',
              minute: '2-digit'
            }
          )
      };


      onUpdateSessionMessages([
        ...updatedMessages,
        botError
      ]);

    } finally {
      setIsProcessing(false);
    }
  };


  /* ========================================================================
   * UI
   * ====================================================================== */

  const samplePrompts = [
    {
      title:
        'Urban Infrastructure',

      text:
        'Identify new building constructions, road expansions, and urban density changes.'
    },

    {
      title:
        'Land Cover & Canopy',

      text:
        'Classify vegetation canopy, water bodies, agricultural parcels, and bare ground.'
    },

    {
      title:
        'Temporal Progression',

      text:
        'Analyze historical satellite sequence to map environmental and morphological shifts.'
    }
  ];


  const messages =
    activeSession?.messages || [];


  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        backgroundColor:
          'var(--bg-primary)',
        position: 'relative'
      }}
    >

      {/* Header */}
      <div
        style={{
          padding:
            '0.9rem 1.75rem',
          borderBottom:
            '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent:
            'space-between',
          background:
            'var(--bg-secondary)'
        }}
      >

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}
        >

          <h2
            style={{
              fontSize: '0.95rem',
              fontWeight: 600,
              color:
                'var(--text-primary)',
              letterSpacing:
                '-0.01em'
            }}
          >
            {activeSession?.title ||
              'New Analysis Session'}
          </h2>

        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        />

      </div>


      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '2rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem'
        }}
      >

        {messages.length === 0 ? (

          <div
            style={{
              margin: 'auto',
              maxWidth: '680px',
              width: '100%',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1.25rem',
              padding: '2rem 1rem'
            }}
          >

            <img
              src={logoSrc}
              alt="AKASHA"
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '12px',
                objectFit: 'contain',
                marginBottom: '4px'
              }}
              draggable="false"
            />

            <div>

              <h3
                style={{
                  fontSize: '1.45rem',
                  fontWeight: 700,
                  letterSpacing:
                    '-0.02em',
                  color:
                    'var(--text-primary)',
                  marginBottom: '6px'
                }}
              >
                Earth Observation Intelligence
              </h3>

              <p
                style={{
                  color:
                    'var(--text-secondary)',
                  fontSize: '0.9rem',
                  lineHeight: 1.5,
                  maxWidth: '520px',
                  margin: '0 auto'
                }}
              >
                Upload satellite imagery and ask queries.
                AKASHA routes your request to specialized models
                like GeoChat, TEOChat, and M2CD.
              </p>

            </div>


            <div
              style={{
                display: 'grid',
                gridTemplateColumns:
                  'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '12px',
                width: '100%',
                marginTop: '0.5rem'
              }}
            >

              {samplePrompts.map(
                (prompt, index) => (

                  <div
                    key={index}
                    onClick={() =>
                      handleSend(
                        prompt.text
                      )
                    }
                    className="glass-interactive"
                    style={{
                      padding:
                        '14px 16px',
                      borderRadius: '10px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      fontSize: '0.84rem'
                    }}
                  >

                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent:
                          'space-between',
                        marginBottom: '6px'
                      }}
                    >

                      <strong
                        style={{
                          color:
                            'var(--text-primary)',
                          fontSize: '0.86rem'
                        }}
                      >
                        {prompt.title}
                      </strong>

                      <ArrowUpRight
                        size={14}
                        color="var(--text-muted)"
                      />

                    </div>

                    <span
                      style={{
                        color:
                          'var(--text-muted)',
                        fontSize: '0.78rem',
                        lineHeight: 1.4,
                        display: 'block'
                      }}
                    >
                      {prompt.text}
                    </span>

                  </div>

                )
              )}

            </div>

          </div>

        ) : (

          messages.map(
            message => {

              const isUser =
                message.sender === 'user';

              return (
                <div
                  key={message.id}
                  style={{
                    display: 'flex',
                    gap: '12px',
                    alignSelf:
                      isUser
                        ? 'flex-end'
                        : 'flex-start',
                    maxWidth:
                      isUser
                        ? '72%'
                        : '84%',
                    animation:
                      'fadeIn 0.25s ease-out'
                  }}
                >

                  {!isUser && (
                    <div
                      style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '8px',
                        background:
                          'var(--bg-card)',
                        border:
                          '1px solid var(--border-medium)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent:
                          'center',
                        flexShrink: 0
                      }}
                    >
                      <Bot
                        size={17}
                        color="var(--text-primary)"
                      />
                    </div>
                  )}

                  <div
                    style={{
                      display: 'flex',
                      flexDirection:
                        'column',
                      gap: '6px',
                      width: '100%'
                    }}
                  >

                    <div
                      style={{
                        padding:
                          '12px 16px',
                        borderRadius:
                          isUser
                            ? '14px 14px 2px 14px'
                            : '14px 14px 14px 2px',
                        background:
                          isUser
                            ? 'var(--bg-card)'
                            : 'var(--bg-secondary)',
                        border:
                          isUser
                            ? '1px solid var(--border-medium)'
                            : '1px solid var(--border-subtle)',
                        color:
                          'var(--text-primary)',
                        boxShadow:
                          '0 2px 8px rgba(0,0,0,0.06)'
                      }}
                    >

                      {message.attachments &&
                        message.attachments.length >
                          0 && (

                          <div
                            style={{
                              display: 'flex',
                              gap: '8px',
                              flexWrap: 'wrap',
                              marginBottom:
                                '8px'
                            }}
                          >

                            {message.attachments.map(
                              (attachment, index) => {

                                const imageUrl =
                                  typeof attachment ===
                                  'string'
                                    ? attachment
                                    : (
                                      attachment?.url ||
                                      attachment?.preview
                                    );

                                const imageName =
                                  typeof attachment ===
                                  'string'
                                    ? 'Satellite Image'
                                    : (
                                      attachment?.name ||
                                      'Satellite Image'
                                    );

                                return (
                                  <div
                                    key={index}
                                    style={{
                                      display:
                                        'flex',
                                      flexDirection:
                                        'column',
                                      gap: '4px'
                                    }}
                                  >

                                    {imageUrl ? (

                                      <img
                                        src={imageUrl}
                                        alt={imageName}
                                        onClick={() =>
                                          setPreviewImage({
                                            url: imageUrl,
                                            name: imageName
                                          })
                                        }
                                        title="Click to enlarge"
                                        style={{
                                          maxWidth:
                                            '220px',
                                          maxHeight:
                                            '150px',
                                          borderRadius:
                                            '8px',
                                          objectFit:
                                            'cover',
                                          border:
                                            '1px solid var(--border-subtle)',
                                          cursor:
                                            'pointer',
                                          transition:
                                            'border-color 0.15s ease'
                                        }}
                                        onMouseEnter={event => {
                                          event.currentTarget.style.borderColor =
                                            'var(--border-strong)';
                                        }}
                                        onMouseLeave={event => {
                                          event.currentTarget.style.borderColor =
                                            'var(--border-subtle)';
                                        }}
                                      />

                                    ) : (

                                      <span
                                        style={{
                                          display:
                                            'inline-flex',
                                          alignItems:
                                            'center',
                                          gap: '4px',
                                          background:
                                            'var(--bg-card-hover)',
                                          border:
                                            '1px solid var(--border-subtle)',
                                          padding:
                                            '4px 8px',
                                          borderRadius:
                                            '6px',
                                          fontSize:
                                            '0.74rem',
                                          color:
                                            'var(--text-primary)'
                                        }}
                                      >

                                        <ImageIcon size={12} />

                                        {imageName}

                                      </span>

                                    )}

                                  </div>
                                );
                              }
                            )}

                          </div>
                        )}

                      <p
                        style={{
                          fontSize:
                            '0.92rem',
                          lineHeight:
                            1.55,
                          whiteSpace:
                            'pre-wrap',
                          color:
                            message.isError
                              ? '#ef4444'
                              : 'var(--text-primary)'
                        }}
                      >
                        {message.text}
                      </p>

                    </div>


                    {message.timestamp && (
                      <div
                        style={{
                          fontSize:
                            '0.72rem',
                          color:
                            'var(--text-muted)',
                          textAlign:
                            isUser
                              ? 'right'
                              : 'left',
                          paddingLeft:
                            isUser
                              ? '0'
                              : '2px',
                          paddingRight:
                            isUser
                              ? '2px'
                              : '0'
                        }}
                      >
                        <span>
                          {message.timestamp}
                        </span>
                      </div>
                    )}

                  </div>


                  {isUser && (
                    <div
                      style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '8px',
                        background:
                          'var(--bg-card)',
                        border:
                          '1px solid var(--border-medium)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent:
                          'center',
                        flexShrink: 0
                      }}
                    >
                      <User
                        size={16}
                        color="var(--text-secondary)"
                      />
                    </div>
                  )}

                </div>
              );
            }
          )
        )}


        {isProcessing && (
          <div
            style={{
              display: 'flex',
              gap: '12px',
              alignItems: 'center'
            }}
          >

            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background:
                  'var(--bg-card)',
                border:
                  '1px solid var(--border-medium)',
                display: 'flex',
                alignItems: 'center',
                justifyContent:
                  'center'
              }}
            >

              <Bot
                size={17}
                color="var(--text-primary)"
              />

            </div>


            <div
              style={{
                padding:
                  '10px 16px',
                borderRadius: '12px',
                background:
                  'var(--bg-card)',
                border:
                  '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}
            >

              <div className="spinner" />

              <span
                style={{
                  fontSize:
                    '0.86rem',
                  color:
                    'var(--text-secondary)'
                }}
              >
                Processing imagery with specialist model...
              </span>

            </div>

          </div>
        )}

        <div ref={messagesEndRef} />

      </div>


      {/* Input */}
      <div
        style={{
          padding:
            '1rem 1.75rem 1.25rem 1.75rem',
          background:
            'var(--bg-secondary)',
          borderTop:
            '1px solid var(--border-subtle)'
        }}
      >

        {selectedFiles &&
          selectedFiles.length > 0 && (

          <div
            style={{
              display: 'flex',
              gap: '6px',
              overflowX: 'auto',
              paddingBottom: '8px',
              marginBottom: '4px'
            }}
          >

            {selectedFiles.map(
              (file, index) => (

                <div
                  key={index}
                  style={{
                    display:
                      'inline-flex',
                    alignItems:
                      'center',
                    gap: '6px',
                    background:
                      'var(--bg-card)',
                    border:
                      '1px solid var(--border-medium)',
                    borderRadius:
                      '6px',
                    padding:
                      '3px 8px',
                    fontSize:
                      '0.76rem',
                    color:
                      'var(--text-primary)'
                  }}
                >

                  <ImageIcon
                    size={13}
                    color="var(--text-secondary)"
                  />

                  <span>
                    {file.name}
                  </span>

                  <X
                    size={13}
                    style={{
                      cursor: 'pointer',
                      marginLeft: '4px',
                      color:
                        'var(--text-muted)'
                    }}
                    onClick={() =>
                      onFileSelect(
                        selectedFiles.filter(
                          (_, itemIndex) =>
                            itemIndex !==
                            index
                        )
                      )
                    }
                  />

                </div>
              )
            )}

          </div>
        )}


        <div
          style={{
            display: 'flex',
            gap: '8px',
            alignItems: 'center'
          }}
        >

          <label
            className="btn-secondary"
            style={{
              padding: '10px',
              borderRadius: '8px',
              cursor: 'pointer'
            }}
            title="Attach Satellite Imagery"
          >

            <input
              type="file"
              multiple
              accept=".tiff,.tif,.png,.jpeg,.jpg"
              style={{
                display: 'none'
              }}
              onChange={event => {
                const incomingFiles =
                  Array.from(
                    event.target.files ||
                    []
                  );

                if (
                  incomingFiles.length === 0
                ) {
                  event.target.value = '';
                  return;
                }

                const combined = [
                  ...(selectedFiles || []),
                  ...incomingFiles
                ];

                if (
                  combined.length >
                  MAX_UPLOAD_FILES
                ) {
                  console.warn(
                    `Upload capped at ${MAX_UPLOAD_FILES} files.`
                  );
                }

                onFileSelect(
                  combined.slice(
                    0,
                    MAX_UPLOAD_FILES
                  )
                );

                event.target.value = '';
              }}
            />

            <Paperclip size={17} />

          </label>


          <input
            type="text"
            className="glass-input"
            placeholder="Ask about imagery features, segmentation, or morphological changes..."
            value={query}
            onChange={event =>
              setQuery(
                event.target.value
              )
            }
            onKeyDown={event =>
              event.key === 'Enter' &&
              handleSend()
            }
            disabled={
              isProcessing
            }
            style={{
              borderRadius: '8px',
              padding:
                '11px 14px'
            }}
          />


          <button
            className="btn-primary"
            onClick={() =>
              handleSend()
            }
            disabled={
              (
                (
                  !query.trim()
                ) &&
                (
                  !selectedFiles ||
                  selectedFiles.length ===
                    0
                )
              ) ||
              isProcessing
            }
            style={{
              padding:
                '10px 18px',
              borderRadius: '8px'
            }}
          >

            {isProcessing ? (
              <div className="spinner" />
            ) : (
              <Send size={16} />
            )}

            <span>
              Run
            </span>

          </button>

        </div>

      </div>


      {/* Lightbox */}
      {previewImage && (
        <div
          onClick={() =>
            setPreviewImage(null)
          }
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            background:
              'rgba(0, 0, 0, 0.88)',
            backdropFilter:
              'blur(8px)',
            display: 'flex',
            alignItems:
              'center',
            justifyContent:
              'center',
            padding: '2rem',
            animation:
              'fadeIn 0.2s ease-out'
          }}
        >

          <div
            onClick={event =>
              event.stopPropagation()
            }
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection:
                'column',
              background:
                'var(--bg-card)',
              border:
                '1px solid var(--border-medium)',
              borderRadius:
                '12px',
              padding: '1.25rem',
              boxShadow:
                '0 24px 60px rgba(0,0,0,0.5)'
            }}
          >

            <div
              style={{
                display: 'flex',
                width: '100%',
                alignItems: 'center',
                justifyContent:
                  'space-between',
                marginBottom:
                  '0.85rem',
                gap: '12px'
              }}
            >

              <h4
                style={{
                  color:
                    'var(--text-primary)',
                  fontSize:
                    '0.92rem',
                  fontWeight: 600
                }}
              >
                {previewImage.name ||
                  'Satellite Imagery'}
              </h4>


              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  alignItems:
                    'center'
                }}
              >

                <a
                  href={
                    previewImage.url
                  }
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary"
                  style={{
                    fontSize:
                      '0.76rem',
                    padding:
                      '5px 10px'
                  }}
                >
                  Open Original
                </a>


                <button
                  onClick={() =>
                    setPreviewImage(null)
                  }
                  style={{
                    background:
                      'transparent',
                    border: 'none',
                    color:
                      'var(--text-primary)',
                    cursor: 'pointer',
                    padding: '4px',
                    display: 'flex'
                  }}
                >
                  <X size={18} />
                </button>

              </div>

            </div>


            <img
              src={
                previewImage.url
              }
              alt={
                previewImage.name ||
                'Full preview'
              }
              style={{
                maxWidth: '100%',
                maxHeight: '75vh',
                borderRadius: '8px',
                objectFit: 'contain',
                border:
                  '1px solid var(--border-subtle)'
              }}
            />

          </div>

        </div>
      )}

    </div>
  );
};


export default ChatInterface;
