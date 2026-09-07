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
import { auth, storage, db, isDemoMode } from '../firebaseClient';
import { ref, uploadBytes } from 'firebase/storage';
import { collection, addDoc } from 'firebase/firestore';

const MAX_UPLOAD_FILES = 4;
const SUPPORTED_EXTENSIONS = new Set([
  '.tif',
  '.tiff',
  '.png',
  '.jpg',
  '.jpeg'
]);

/* -------------------------------------------------------------------------- */
/* File / remote-sensing metadata helpers                                     */
/* -------------------------------------------------------------------------- */

const getExtension = (filename = '') => {
  const match = filename.toLowerCase().match(/(\.[a-z0-9]+)$/);
  return match ? match[1] : '';
};

const getBaseName = (filename = '') => {
  const extension = getExtension(filename);
  return extension
    ? filename.slice(0, -extension.length)
    : filename;
};

const sanitizeIdPart = (value = '') =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');

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
    lower.includes('_vv_') ||
    lower.includes('_vh_') ||
    /(^|[_-])(vv|vh)(?=[_.-]|$)/i.test(lower);

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

  // For common image formats, defaulting to optical is reasonable.
  // For TIFF/GeoTIFF, do not silently misclassify an unknown scientific raster.
  const extension = getExtension(filename);

  if (extension === '.png' || extension === '.jpg' || extension === '.jpeg') {
    return 'optical';
  }

  return null;
};

const inferPolarization = (filename = '') => {
  const lower = filename.toLowerCase();

  // Find all standalone VV/VH tokens and use the last one.
  // This handles names such as:
  // ...VV_VH_VH_decibel_gamma0.tiff
  // ...VV_VH_VV_decibel_gamma0.tiff
  const matches = [
    ...lower.matchAll(/(?:^|[_-])(vv|vh)(?=[_.-]|$)/gi)
  ];

  if (matches.length === 0) {
    return null;
  }

  return matches[matches.length - 1][1].toUpperCase();
};

const inferAcquisitionTime = (filename = '') => {
  /*
   * Supports common filename fragments such as:
   *
   * 2024-01-04-00_00_2024-01-04-23_59_...
   * 2026-08-25-00_00_2026-08-25-23_59_...
   *
   * The first YYYY-MM-DD occurrence is used as acquisition date.
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

const inferObservationSeed = (filename = '') => {
  const modality = inferModality(filename);
  const date = inferAcquisitionTime(filename);
  const polarization = inferPolarization(filename);

  if (modality === 'sar') {
    return [
      'sar',
      date || 'unknown_date',
      'same_acquisition',
      sanitizeIdPart(
        getBaseName(filename)
          .replace(
            /(?:^|[_-])(?:vv|vh)(?=[_.-]|$)/gi,
            ''
          )
      )
    ].join('_');
  }

  return [
    modality || 'unknown',
    date || 'unknown_date',
    sanitizeIdPart(getBaseName(filename))
  ].join('_');
};

const buildUploadDescriptors = (files) => {
  const sarGroups = new Map();

  return files.map((file, index) => {
    const filename = file.name || `image_${index + 1}.tiff`;
    const extension = getExtension(filename);

    if (!SUPPORTED_EXTENSIONS.has(extension)) {
      throw new Error(
        `Unsupported imagery format: ${filename}`
      );
    }

    const modality = inferModality(filename);

    if (!modality) {
      throw new Error(
        `Could not determine whether "${filename}" is optical or SAR. ` +
        `Use a recognizable remote-sensing filename or provide imagery metadata.`
      );
    }

    const acquisitionTime = inferAcquisitionTime(filename);
    const polarization =
      modality === 'sar'
        ? inferPolarization(filename)
        : null;

    if (modality === 'sar' && !polarization) {
      throw new Error(
        `Could not determine VV/VH polarization from "${filename}". ` +
        `A SAR observation requires both VV and VH files.`
      );
    }

    let observationId;

    if (modality === 'sar') {
      const sarSeed = [
        acquisitionTime || 'unknown_date',
        sanitizeIdPart(
          filename
            .replace(
              /(?:^|[_-])(?:vv|vh)(?=[_.-]|$)/gi,
              ''
            )
        )
      ].join('_');

      if (!sarGroups.has(sarSeed)) {
        sarGroups.set(
          sarSeed,
          `obs_sar_${sanitizeIdPart(sarSeed)}`
        );
      }

      observationId = sarGroups.get(sarSeed);
    } else {
      observationId =
        `obs_${inferObservationSeed(filename)}_${index}`;
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
};

const validateUploadCombination = (descriptors) => {
  if (descriptors.length === 0) {
    return;
  }

  const sarDescriptors = descriptors.filter(
    item => item.modality === 'sar'
  );

  const opticalDescriptors = descriptors.filter(
    item => item.modality === 'optical'
  );

  // A logical SAR observation requires exactly one VV + one VH.
  const sarGroups = new Map();

  for (const item of sarDescriptors) {
    if (!sarGroups.has(item.observationId)) {
      sarGroups.set(item.observationId, []);
    }

    sarGroups.get(item.observationId).push(item);
  }

  for (const [observationId, group] of sarGroups.entries()) {
    const polarizations = group.map(
      item => item.polarization
    );

    const vvCount = polarizations.filter(p => p === 'VV').length;
    const vhCount = polarizations.filter(p => p === 'VH').length;

    if (vvCount !== 1 || vhCount !== 1) {
      throw new Error(
        `SAR observation "${observationId}" must contain exactly one VV file ` +
        `and one VH file.`
      );
    }
  }

  /*
   * Physical upload count is capped at four.
   *
   * Examples:
   *   optical                     = 1 file
   *   SAR                         = 2 files
   *   optical + SAR               = 3 files
   *   SAR + SAR                   = 4 files
   *   optical + optical           = 2 files
   */
  if (descriptors.length > MAX_UPLOAD_FILES) {
    throw new Error(
      `You can upload at most ${MAX_UPLOAD_FILES} files.`
    );
  }

  // This is primarily a sanity check. The backend remains authoritative.
  if (sarDescriptors.length > 0 && opticalDescriptors.length > 2) {
    throw new Error(
      'The selected imagery exceeds the supported physical-file configuration.'
    );
  }

  return {
    sarCount: sarGroups.size,
    opticalCount: opticalDescriptors.length
  };
};

/* -------------------------------------------------------------------------- */
/* Component                                                                  */
/* -------------------------------------------------------------------------- */

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
  const query = draftQuery || '';
  const setQuery = onDraftQueryChange;

  const [isProcessing, setIsProcessing] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth'
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, isProcessing]);

  /* ------------------------------------------------------------------------ */
  /* Send                                                                     */
  /* ------------------------------------------------------------------------ */

  const handleSend = async (overrideQuery = null) => {
    let textToSend = overrideQuery || query;

    if (
      (!textToSend || !textToSend.trim()) &&
      selectedFiles &&
      selectedFiles.length > 0
    ) {
      textToSend = 'Analyze attached satellite imagery.';
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
    ).slice(0, MAX_UPLOAD_FILES);

    /* ---------------------------------------------------------------------- */
    /* Build logical observation descriptors BEFORE upload                    */
    /* ---------------------------------------------------------------------- */

    let descriptors = [];

    try {
      descriptors = buildUploadDescriptors(filesToUpload);
      validateUploadCombination(descriptors);
    } catch (validationError) {
      const botError = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        isError: true,
        text: validationError.message,
        timestamp: new Date().toLocaleTimeString(
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
          id: Date.now().toString(),
          sender: 'user',
          text: textToSend,
          attachments: filesToUpload.map(file => ({
            name: file.name,
            url: URL.createObjectURL(file)
          })),
          timestamp: new Date().toLocaleTimeString(
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

    /* ---------------------------------------------------------------------- */
    /* Step 1: User message                                                   */
    /* ---------------------------------------------------------------------- */

    const userMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: textToSend,
      attachments: [],
      timestamp: new Date().toLocaleTimeString(
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

    onUpdateSessionMessages(messagesWithUser);

    /* ---------------------------------------------------------------------- */
    /* Step 2: Upload imagery with trusted remote-sensing metadata            */
    /* ---------------------------------------------------------------------- */

    const attachmentUrls = [];
    const uploadedImagePaths = [];
    const uploadedImageMetadata = [];

    try {
      for (const descriptor of descriptors) {
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

        if (user && !isDemoMode) {
          const uid =
            user.uid ||
            user.id ||
            user.firebaseUser?.uid;

          if (!uid) {
            throw new Error(
              'Logged-in user does not have a valid Firebase UID.'
            );
          }

          const timestamp = Date.now();

          storagePath =
            `users/${uid}/imagery/${timestamp}_${file.name}`;

          const storageRef = ref(
            storage,
            storagePath
          );

          /*
           * IMPORTANT:
           * Firebase Storage customMetadata is what the backend can retrieve
           * as trusted metadata later when building the input manifest.
           */
          const customMetadata = {
            original_filename: filename,
            file_format: extension.replace('.', '').toLowerCase(),
            modality,
            acquisition_time:
              acquisitionTime || '',
            observation_id: observationId,
            polarization:
              polarization || '',
            imagery_role:
              modality === 'sar'
                ? 'sar_channel'
                : 'single_image',
            uploaded_by:
              'akasha_frontend',
            metadata_version:
              '2'
          };

          await uploadBytes(
            storageRef,
            file,
            {
              customMetadata
            }
          );

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
              uploadedAt: Date.now(),
              size: file.size,

              // Store the same normalized metadata in Firestore.
              modality,
              polarization:
                polarization || null,
              acquisitionTime:
                acquisitionTime || null,
              observationId,
              format: extension
                .replace('.', '')
                .toLowerCase()
            }
          );

          uploadedImagePaths.push(storagePath);

          uploadedImageMetadata.push({
            path: storagePath,
            filename,
            format: extension
              .replace('.', '')
              .toLowerCase(),
            modality,
            polarization:
              polarization || null,
            acquisition_time:
              acquisitionTime || null,
            observation_id: observationId
          });

          localPreviewUrl =
            URL.createObjectURL(file);
        } else {
          /*
           * Demo mode remains local-only.
           * The backend analysis call below is still blocked unless the
           * application is configured for demo analysis.
           */
          localPreviewUrl =
            URL.createObjectURL(file);

          uploadedImageMetadata.push({
            path: null,
            filename,
            format: extension
              .replace('.', '')
              .toLowerCase(),
            modality,
            polarization:
              polarization || null,
            acquisition_time:
              acquisitionTime || null,
            observation_id: observationId
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
        attachments: attachmentUrls
      };

      const updatedMessages =
        messagesWithUser.map(m =>
          m.id === userMessage.id
            ? userMessageWithAttachments
            : m
        );

      onUpdateSessionMessages([
        ...updatedMessages,
        {
          id: (Date.now() + 1).toString(),
          sender: 'assistant',
          isError: true,
          text: uploadErrorMessage,
          timestamp: new Date().toLocaleTimeString(
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
      attachments: attachmentUrls
    };

    const updatedMessages =
      messagesWithUser.map(m =>
        m.id === userMessage.id
          ? userMessageWithAttachments
          : m
      );

    onUpdateSessionMessages(
      updatedMessages
    );

    /* ---------------------------------------------------------------------- */
    /* Step 3: Backend analysis request                                       */
    /* ---------------------------------------------------------------------- */

    try {
      if (!user && !isDemoMode) {
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
       * `image_paths` remains the physical-file transport layer.
       *
       * `image_metadata` is the normalized description of those physical files.
       * The backend must still verify metadata from Firebase Storage instead
       * of trusting this request body.
       */
      const requestBody = {
        query: textToSend,

        // Backward-compatible first physical file.
        image_path:
          uploadedImagePaths[0] || null,

        // All physical uploaded files, max four.
        image_paths:
          uploadedImagePaths,

        // Explicit client-side observation hints.
        image_metadata:
          uploadedImageMetadata,

        max_new_tokens: 256
      };

      const response = await axios.post(
        '/api/analyze',
        requestBody,
        {
          timeout: 30000,
          headers: {
            Authorization:
              `Bearer ${idToken}`
          }
        }
      );

      const data = response.data;

      const botMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text:
          data.answer ||
          'Analysis completed without a textual response.',
        timestamp: new Date().toLocaleTimeString(
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

      const isOffline =
        !error?.response;

      let errorMessage;

      if (
        detail.includes('ZeroGPU') ||
        detail.includes('runs limit') ||
        detail.includes('daily AI analysis limit')
      ) {
        errorMessage =
          'Your daily AI analysis limit has been reached. Please try again later.';
      } else if (
        status === 400 &&
        detail
      ) {
        errorMessage =
          `Analysis failed (400): ${detail}`;
      } else if (isOffline) {
        errorMessage =
          'Backend offline. Ensure the Python API service is running on port 8000.';
      } else {
        errorMessage =
          `Analysis failed (${status || 'error'}): ${
            detail ||
            error?.message ||
            'Please try again.'
          }`;
      }

      const botError = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        isError: true,
        text: errorMessage,
        timestamp: new Date().toLocaleTimeString(
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

  /* ------------------------------------------------------------------------ */
  /* UI                                                                       */
  /* ------------------------------------------------------------------------ */

  const samplePrompts = [
    {
      title: 'Urban Infrastructure',
      text: 'Identify new building constructions, road expansions, and urban density changes.'
    },
    {
      title: 'Land Cover & Canopy',
      text: 'Classify vegetation canopy, water bodies, agricultural parcels, and bare ground.'
    },
    {
      title: 'Temporal Progression',
      text: 'Analyze historical satellite sequence to map environmental and morphological shifts.'
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
      {/* Header Bar */}
      <div
        style={{
          padding: '0.9rem 1.75rem',
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
              letterSpacing: '-0.01em'
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

      {/* Messages Feed */}
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
                Upload satellite imagery and ask
                queries. AKASHA routes your request
                to specialized models like GeoChat,
                TEOChat, and M2CD.
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
                (p, idx) => (
                  <div
                    key={idx}
                    onClick={() =>
                      handleSend(p.text)
                    }
                    className="glass-interactive"
                    style={{
                      padding: '14px 16px',
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
                        {p.title}
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
                      {p.text}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>
        ) : (
          messages.map(msg => {
            const isUser =
              msg.sender === 'user';

            return (
              <div
                key={msg.id}
                style={{
                  display: 'flex',
                  gap: '12px',
                  alignSelf: isUser
                    ? 'flex-end'
                    : 'flex-start',
                  maxWidth: isUser
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
                    flexDirection: 'column',
                    gap: '6px',
                    width: '100%'
                  }}
                >
                  <div
                    style={{
                      padding:
                        '12px 16px',
                      borderRadius: isUser
                        ? '14px 14px 2px 14px'
                        : '14px 14px 14px 2px',
                      background: isUser
                        ? 'var(--bg-card)'
                        : 'var(--bg-secondary)',
                      border: isUser
                        ? '1px solid var(--border-medium)'
                        : '1px solid var(--border-subtle)',
                      color:
                        'var(--text-primary)',
                      boxShadow:
                        '0 2px 8px rgba(0,0,0,0.06)'
                    }}
                  >
                    {msg.attachments &&
                      msg.attachments.length >
                        0 && (
                        <div
                          style={{
                            display: 'flex',
                            gap: '8px',
                            flexWrap:
                              'wrap',
                            marginBottom:
                              '8px'
                          }}
                        >
                          {msg.attachments.map(
                            (att, i) => {
                              const imgUrl =
                                typeof att ===
                                'string'
                                  ? att
                                  : (
                                      att?.url ||
                                      att?.preview
                                    );

                              const imgName =
                                typeof att ===
                                'string'
                                  ? 'Satellite Image'
                                  : (
                                      att?.name ||
                                      'Satellite Image'
                                    );

                              return (
                                <div
                                  key={i}
                                  style={{
                                    display:
                                      'flex',
                                    flexDirection:
                                      'column',
                                    gap: '4px'
                                  }}
                                >
                                  {imgUrl ? (
                                    <img
                                      src={imgUrl}
                                      alt={
                                        imgName
                                      }
                                      onClick={() =>
                                        setPreviewImage(
                                          {
                                            url: imgUrl,
                                            name: imgName
                                          }
                                        )
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
                                      onMouseEnter={e =>
                                        e.currentTarget.style.borderColor =
                                          'var(--border-strong)'
                                      }
                                      onMouseLeave={e =>
                                        e.currentTarget.style.borderColor =
                                          'var(--border-subtle)'
                                      }
                                    />
                                  ) : (
                                    <span
                                      style={{
                                        display:
                                          'inline-flex',
                                        alignItems:
                                          'center',
                                        gap:
                                          '4px',
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
                                      <ImageIcon
                                        size={
                                          12
                                        }
                                      />
                                      {
                                        imgName
                                      }
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
                        color: msg.isError
                          ? '#ef4444'
                          : 'var(--text-primary)'
                      }}
                    >
                      {msg.text}
                    </p>
                  </div>

                  {msg.timestamp && (
                    <div
                      style={{
                        fontSize:
                          '0.72rem',
                        color:
                          'var(--text-muted)',
                        textAlign: isUser
                          ? 'right'
                          : 'left',
                        paddingLeft: isUser
                          ? '0'
                          : '2px',
                        paddingRight: isUser
                          ? '2px'
                          : '0'
                      }}
                    >
                      <span>
                        {msg.timestamp}
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
                        '1px solid var(--border-subtle)',
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
          })
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
                  '1px solid var(--border-subtle)',
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
                padding: '10px 16px',
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
                  fontSize: '0.86rem',
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

      {/* Input Area */}
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
        {/* Attachment Chips */}
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
                (file, idx) => (
                  <div
                    key={idx}
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
                      borderRadius: '6px',
                      padding: '3px 8px',
                      fontSize: '0.76rem',
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
                        cursor:
                          'pointer',
                        marginLeft:
                          '4px',
                        color:
                          'var(--text-muted)'
                      }}
                      onClick={() =>
                        onFileSelect(
                          selectedFiles.filter(
                            (_, i) =>
                              i !== idx
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
          {/* File Attachment Button */}
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
              onChange={e => {
                const incomingFiles =
                  Array.from(
                    e.target.files || []
                  );

                if (
                  incomingFiles.length ===
                  0
                ) {
                  e.target.value = '';
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

                e.target.value = '';
              }}
            />

            <Paperclip size={17} />
          </label>

          {/* Text Input */}
          <input
            type="text"
            className="glass-input"
            placeholder="Ask about imagery features, segmentation, or morphological changes..."
            value={query}
            onChange={e =>
              setQuery(
                e.target.value
              )
            }
            onKeyDown={e =>
              e.key === 'Enter' &&
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

          {/* Send Button */}
          <button
            className="btn-primary"
            onClick={() =>
              handleSend()
            }
            disabled={
              (
                (!query.trim() &&
                  (!selectedFiles ||
                    selectedFiles.length ===
                      0)) ||
                isProcessing
              )
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

            <span>Run</span>
          </button>
        </div>
      </div>

      {/* Full-screen Lightbox Modal */}
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
            onClick={e =>
              e.stopPropagation()
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
              borderRadius: '12px',
              padding: '1.25rem',
              boxShadow:
                '0 24px 60px rgba(0,0,0,0.5)'
            }}
          >
            <div
              style={{
                display: 'flex',
                width: '100%',
                alignItems:
                  'center',
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
                {
                  previewImage.name ||
                  'Satellite Imagery'
                }
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
                    setPreviewImage(
                      null
                    )
                  }
                  style={{
                    background:
                      'transparent',
                    border: 'none',
                    color:
                      'var(--text-primary)',
                    cursor:
                      'pointer',
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
                maxWidth:
                  '100%',
                maxHeight:
                  '75vh',
                borderRadius:
                  '8px',
                objectFit:
                  'contain',
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