"""Lightweight internationalization system."""

from __future__ import annotations

import re
from typing import Any

from . import en


class I18n:
    """
    Lightweight localization class.

    Supports:
    - Nested key access: "training.src_dir.label"
    - Variable interpolation: "Found {count} files"
    - Fallback to default locale
    - Runtime locale switching

    Example:
        >>> i18n = I18n(locale="en")
        >>> i18n.t("training.src_dir.label")
        'Source Directory'
        >>> i18n.t("status.files_found", count=42)
        'Found 42 files'
    """

    # Pattern for variable interpolation: {var_name}
    _INTERPOLATE_PATTERN = re.compile(r"\{(\w+)\}")

    def __init__(
        self,
        locale: str = "en",
        fallback_locale: str = "en",
    ) -> None:
        """
        Initialize i18n with specified locale.

        Args:
            locale: Current locale code (e.g., "en", "tr")
            fallback_locale: Fallback locale if key not found
        """
        self._locale = locale
        self._fallback_locale = fallback_locale
        self._translations: dict[str, dict[str, Any]] = {}
        self._load_translations()

    def _load_translations(self) -> None:
        """Load translation dictionaries."""
        self._translations["en"] = en.TRANSLATIONS

        if self._locale != "en":
            try:
                # Dynamic import of locale module
                locale_module = __import__(
                    f"visagen.gui.i18n.{self._locale}", fromlist=["TRANSLATIONS"]
                )
                self._translations[self._locale] = locale_module.TRANSLATIONS
            except ImportError:
                pass  # Fall back to default

    @property
    def locale(self) -> str:
        """Get current locale."""
        return self._locale

    @locale.setter
    def locale(self, value: str) -> None:
        """Set locale and reload translations."""
        self._locale = value
        self._load_translations()

    @property
    def available_locales(self) -> list[str]:
        """List available locale codes."""
        return list(self._translations.keys())

    def t(self, key: str, **kwargs: Any) -> str:
        """
        Translate a key with optional interpolation.

        Args:
            key: Dot-separated translation key (e.g., "training.src_dir.label")
            **kwargs: Variables for interpolation

        Returns:
            Translated string, or key if not found.

        Example:
            >>> i18n.t("errors.file_count", count=5)
            'Found 5 files'
        """
        # Try current locale first
        result = self._get_nested(self._translations.get(self._locale, {}), key)

        # Fall back to default locale
        if result is None and self._locale != self._fallback_locale:
            result = self._get_nested(
                self._translations.get(self._fallback_locale, {}), key
            )

        # Return key itself if not found
        if result is None:
            return key

        # Interpolate variables
        if kwargs:
            result = self._interpolate(result, kwargs)

        return result

    def _get_nested(self, data: dict[str, Any], key: str) -> str | None:
        """
        Get nested value from dictionary using dot notation.

        Args:
            data: Dictionary to search
            key: Dot-separated key path

        Returns:
            Value if found, None otherwise.
        """
        parts = key.split(".")
        current = data

        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None

        return str(current) if current is not None else None

    def _interpolate(self, template: str, values: dict[str, Any]) -> str:
        """
        Replace {var} placeholders with values.

        Args:
            template: String with {var} placeholders
            values: Variable name to value mapping

        Returns:
            Interpolated string.
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return str(values.get(var_name, match.group(0)))

        return self._INTERPOLATE_PATTERN.sub(replacer, template)

    def section(self, prefix: str) -> I18nSection:
        """
        Create a scoped section for convenience.

        Args:
            prefix: Key prefix for all translations in section

        Returns:
            Scoped I18n accessor.

        Example:
            >>> training = i18n.section("training")
            >>> training.t("src_dir.label")  # Same as i18n.t("training.src_dir.label")
        """
        return I18nSection(self, prefix)


class I18nSection:
    """Scoped i18n accessor with a key prefix."""

    def __init__(self, i18n: I18n, prefix: str) -> None:
        self._i18n = i18n
        self._prefix = prefix

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate with prefix prepended."""
        full_key = f"{self._prefix}.{key}"
        return self._i18n.t(full_key, **kwargs)
